"""
AI Article Outline Generator for Hugging Face Spaces
Fixed version v2.2 - Free tier compatible models
"""

import streamlit as st
import json
import re
import os
from typing import Dict, Optional, List, Tuple
from huggingface_hub import InferenceClient
import time


class HuggingFaceOutlineGenerator:
    """Generate article outlines using Hugging Face Inference API."""
    
    # Models confirmed to work with HF Serverless API free tier
    # These are smaller, more accessible models
    MODELS = [
        "google/flan-t5-base",              # Reliable, always available
        "google/flan-t5-large",             # Good quality
        "tiiuae/falcon-7b-instruct",        # If available
        "HuggingFaceH4/zephyr-7b-beta",     # Try if available
    ]
    
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.environ.get("HF_TOKEN")
        self.model_name = None
        self.client = None
        self.working_model_found = False
        
    def initialize_client(self):
        """Initialize the Hugging Face client with error handling."""
        if not self.api_token:
            raise ValueError("Hugging Face token required. Get it from https://huggingface.co/settings/tokens")
    
        try:
            self.client = InferenceClient(token=self.api_token)
            self._find_working_model()
            
        except Exception as e:
            st.error(f"Failed to initialize client: {str(e)}")
            raise
    
    def _find_working_model(self):
        """Test models to find one that works with Serverless Inference API."""
        if self.working_model_found:
            return
        
        st.info("🔍 Testing models to find available one...")
        
        # Track if we saw any 404s vs auth errors
        all_404 = True
        saw_auth_error = False
        
        for model in self.MODELS:
            try:
                st.text(f"Testing {model.split('/')[-1]}...")
                
                # Quick test with minimal tokens
                response = self.client.text_generation(
                    prompt="Hello",
                    model=model,
                    max_new_tokens=5
                )
                
                # If we get here without exception, the model works!
                self.model_name = model
                self.working_model_found = True
                st.success(f"✅ Successfully connected to model: {model.split('/')[-1]}")
                return
                
            except Exception as e:
                error_msg = str(e).lower()
                st.warning(f"❌ {model.split('/')[-1]}: {str(e)[:100]}")
                
                # Track error types
                if "404" in error_msg or "not found" in error_msg:
                    # 404 means model not available, not an auth issue
                    continue
                elif "401" in error_msg or "unauthorized" in error_msg or "invalid token" in error_msg:
                    saw_auth_error = True
                    all_404 = False
                    st.error("🔑 Authentication failed. Your token may be invalid or expired.")
                    st.info("Please:\n1. Go to https://huggingface.co/settings/tokens\n2. Create a NEW token with 'read' permissions\n3. Make sure to copy the FULL token (starts with hf_)")
                    raise ValueError("Invalid token - please get a new HF token")
                else:
                    all_404 = False
                    continue
        
        # If we got here, no model worked
        if all_404:
            st.warning("⚠️ All models returned 404 Not Found. This usually means:")
            st.info("""
            **Possible causes:**
            - Models require accepting terms of use on HuggingFace
            - Models not available on free Serverless tier
            - Models require Pro subscription
            
            **Solution:** The app will use template-based generation, which still works great!
            """)
        else:
            st.warning("⚠️ Could not connect to any model. Using template fallback.")
        
        self.model_name = self.MODELS[0]  # Set default for display

    
    def detect_article_type(self, topic: str) -> str:
        """Detect the type of article from the topic."""
        topic_lower = topic.lower()
        
        if re.search(r'\b(how to|guide|tutorial|step by step)\b', topic_lower):
            return 'how_to'
        elif re.search(r'\b(\d+\s+)?(best|top|vs|versus|comparison)\b', topic_lower):
            return 'listicle'
        elif re.search(r'\b(what is|introduction to|understanding|explain)\b', topic_lower):
            return 'explanatory'
        else:
            return 'general'
    
    def create_prompt(self, topic: str, keyword: str = "") -> str:
        """Create an enhanced prompt for the AI."""
        article_type = self.detect_article_type(topic)
        
        type_guidance = {
            'how_to': "Create a practical how-to guide with clear steps and actionable advice.",
            'listicle': "Create a comparison or list article with detailed reviews and selection criteria.",
            'explanatory': "Create an educational article that explains concepts clearly with examples.",
            'general': "Create a comprehensive article covering all important aspects of the topic."
        }
        
        return f"""You are an expert content strategist and SEO specialist. Generate a detailed, structured article outline.

Topic: {topic}
Target Keyword: {keyword if keyword else "Not specified"}
Article Type: {type_guidance.get(article_type, type_guidance['general'])}

Create a comprehensive article outline with:
1. One compelling main heading (H1) that captures the topic and naturally includes SEO keywords
2. Four well-structured subheadings (H2) that logically break down the topic
3. Three specific, actionable bullet points under each H2
4. Three strategic Call-to-Action (CTA) suggestions placed after relevant sections

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{{
    "h1": "Your SEO-optimized main heading here",
    "sections": [
        {{
            "h2": "First section heading",
            "bullets": [
                "First specific, actionable point with details",
                "Second point with practical guidance",
                "Third point with clear value"
            ]
        }},
        {{
            "h2": "Second section heading",
            "bullets": ["Detailed point 1", "Detailed point 2", "Detailed point 3"]
        }},
        {{
            "h2": "Third section heading",
            "bullets": ["Specific guidance 1", "Specific guidance 2", "Specific guidance 3"]
        }},
        {{
            "h2": "Fourth section heading",
            "bullets": ["Actionable tip 1", "Actionable tip 2", "Actionable tip 3"]
        }}
    ],
    "ctas": [
        {{"after": 0, "text": "Relevant CTA after first section"}},
        {{"after": 1, "text": "Relevant CTA after second section"}},
        {{"after": 3, "text": "Final CTA after last section"}}
    ]
}}

Make the outline SEO-optimized, reader-friendly, and highly specific to the topic.
Respond with ONLY the JSON, no additional text."""
    
    def generate_outline(self, topic: str, keyword: str = "") -> Dict:
        """Generate article outline using Hugging Face Inference API with retry logic."""
        if self.client is None:
            self.initialize_client()
        
        # If no working model was found during initialization, use template
        if not self.working_model_found:
            st.info("📋 Using template-based generation (AI models unavailable)")
            return self._generate_template_based(topic, keyword)
    
        prompt = self.create_prompt(topic, keyword)
    
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response_text = self.client.text_generation(
                    prompt=prompt,
                    model=self.model_name,
                    max_new_tokens=1500,
                    temperature=0.7,
                    return_full_text=False
                )
                
                outline = self._parse_response(response_text, topic, keyword)
                return outline
            
            except Exception as e:
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "429" in error_msg:
                    if attempt < max_retries - 1:
                        st.warning(f"⏳ Rate limit hit, retrying in 5 seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(5)
                        continue
                    else:
                        st.error("⏳ Rate limit reached. Using template fallback.")
                elif "token" in error_msg or "401" in error_msg or "unauthorized" in error_msg:
                    st.error("🔑 Invalid or expired token. Please check your Hugging Face token.")
                    st.info("Get a new token at: https://huggingface.co/settings/tokens")
                elif "404" in error_msg or "not found" in error_msg:
                    st.error("❌ Model temporarily unavailable. Using template fallback.")
                else:
                    st.warning(f"⚠️ {str(e)[:200]}")
                
                # Fallback to template
                return self._generate_template_based(topic, keyword)
        
        # If all retries failed
        return self._generate_template_based(topic, keyword)
    
    def _parse_response(self, response_text: str, topic: str, keyword: str) -> Dict:
        """Parse AI response and extract JSON outline."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group(0)
                outline = json.loads(json_str)
                
                # Validate structure
                if self._validate_outline(outline):
                    return outline
            
            # If parsing failed, create structured outline from text
            return self._create_structured_outline_from_text(response_text, topic, keyword)
            
        except json.JSONDecodeError:
            return self._create_structured_outline_from_text(response_text, topic, keyword)
    
    def _validate_outline(self, outline: Dict) -> bool:
        """Validate outline structure."""
        required_keys = {'h1', 'sections', 'ctas'}
        if not all(key in outline for key in required_keys):
            return False
        
        if not isinstance(outline['sections'], list) or len(outline['sections']) < 3:
            return False
        
        for section in outline['sections']:
            if 'h2' not in section or 'bullets' not in section:
                return False
            if not isinstance(section['bullets'], list) or len(section['bullets']) < 2:
                return False
        
        return True
    
    def _create_structured_outline_from_text(self, text: str, topic: str, keyword: str) -> Dict:
        """Create a structured outline from unstructured AI response."""
        st.warning("⚠️ AI response wasn't in perfect format. Creating structured outline...")
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        sections = []
        current_section = None
        
        for line in lines:
            if any(marker in line.lower() for marker in ['section', 'step', '##', 'h2:', '**']):
                if current_section and len(current_section['bullets']) > 0:
                    sections.append(current_section)
                current_section = {'h2': line.strip('#*: '), 'bullets': []}
            elif current_section and line and not line.startswith('{') and not line.startswith('}'):
                clean_line = line.strip('•-*: ')
                if len(clean_line) > 10:
                    current_section['bullets'].append(clean_line)
        
        if current_section and len(current_section['bullets']) > 0:
            sections.append(current_section)
        
        if len(sections) < 3:
            return self._generate_template_based(topic, keyword)
        
        h1 = f"Complete Guide to {topic.title()}"
        if keyword:
            h1 = f"{topic.title()}: {keyword.title()}"
        
        for section in sections:
            while len(section['bullets']) < 3:
                section['bullets'].append(f"Additional details about {section['h2'].lower()}")
        
        sections = sections[:4]
        
        ctas = [
            {"after": 0, "text": "Ready to get started? Learn more about implementing these strategies."},
            {"after": 1, "text": "Want expert guidance? Check out our comprehensive resources."},
            {"after": len(sections)-1, "text": "Start applying these tips today for best results!"}
        ]
        
        return {
            'h1': h1,
            'sections': sections,
            'ctas': ctas
        }
    
    def _generate_template_based(self, topic: str, keyword: str = "") -> Dict:
        """Generate outline using template when AI is unavailable."""
        article_type = self.detect_article_type(topic)
        
        templates = {
            'how_to': {
                'h1': f"How to {topic.replace('how to', '').strip().title()}: Complete Step-by-Step Guide",
                'sections': [
                    {
                        'h2': 'Getting Started: Prerequisites and Preparation',
                        'bullets': [
                            'Understanding the basic requirements and tools needed',
                            'Setting up your environment for success',
                            'Common mistakes to avoid before starting'
                        ]
                    },
                    {
                        'h2': 'Step-by-Step Implementation Process',
                        'bullets': [
                            'Following the proven methodology step by step',
                            'Best practices and expert tips for each phase',
                            'Troubleshooting common issues as you progress'
                        ]
                    },
                    {
                        'h2': 'Advanced Techniques and Optimization',
                        'bullets': [
                            'Taking your results to the next level',
                            'Professional strategies used by experts',
                            'Measuring and improving your outcomes'
                        ]
                    },
                    {
                        'h2': 'Maintaining Long-Term Success',
                        'bullets': [
                            'Ongoing maintenance and continuous improvement',
                            'Scaling your approach as you grow',
                            'Resources for continued learning and support'
                        ]
                    }
                ]
            },
            'listicle': {
                'h1': f"Top Picks for {topic.title()}: Expert Comparison and Review",
                'sections': [
                    {
                        'h2': 'Selection Criteria: What Makes the Best Choice',
                        'bullets': [
                            'Key factors to consider when evaluating options',
                            'Understanding your specific needs and priorities',
                            'Quality indicators and red flags to watch for'
                        ]
                    },
                    {
                        'h2': 'Top-Rated Options: Detailed Comparison',
                        'bullets': [
                            'In-depth analysis of leading choices',
                            'Pros and cons of each option',
                            'Real-world performance and user experiences'
                        ]
                    },
                    {
                        'h2': 'Value Analysis: Finding the Best Deal',
                        'bullets': [
                            'Price comparison and value for money assessment',
                            'Hidden costs and long-term investment considerations',
                            'Best options for different budgets'
                        ]
                    },
                    {
                        'h2': 'Making Your Decision: Final Recommendations',
                        'bullets': [
                            'Best overall choice for most users',
                            'Alternative recommendations for specific use cases',
                            'Where to buy and what to watch out for'
                        ]
                    }
                ]
            },
            'explanatory': {
                'h1': f"Understanding {topic.title()}: A Comprehensive Explanation",
                'sections': [
                    {
                        'h2': 'The Fundamentals: Core Concepts Explained',
                        'bullets': [
                            'Breaking down the basic principles in simple terms',
                            'Key terminology and definitions you need to know',
                            'Historical context and why it matters today'
                        ]
                    },
                    {
                        'h2': 'How It Works: The Mechanics Behind the Concept',
                        'bullets': [
                            'Step-by-step explanation of the process',
                            'Real-world examples that illustrate the concept',
                            'Common misconceptions and clarifications'
                        ]
                    },
                    {
                        'h2': 'Practical Applications and Use Cases',
                        'bullets': [
                            'Where and when this concept applies in real life',
                            'Benefits and advantages of understanding it',
                            'Common scenarios where this knowledge is useful'
                        ]
                    },
                    {
                        'h2': 'Going Deeper: Advanced Insights',
                        'bullets': [
                            'More complex aspects for those ready to learn more',
                            'Related concepts and further areas of study',
                            'Resources for continued learning and exploration'
                        ]
                    }
                ]
            },
            'general': {
                'h1': f"{topic.title()}: Essential Information and Insights",
                'sections': [
                    {
                        'h2': 'Introduction and Overview',
                        'bullets': [
                            f'Understanding the basics of {topic}',
                            'Why this topic is relevant and important',
                            'What you\'ll learn in this comprehensive guide'
                        ]
                    },
                    {
                        'h2': 'Key Components and Elements',
                        'bullets': [
                            'Breaking down the main aspects and features',
                            'How different elements work together',
                            'Critical factors that influence outcomes'
                        ]
                    },
                    {
                        'h2': 'Practical Implementation and Best Practices',
                        'bullets': [
                            'Proven strategies and approaches that work',
                            'Common pitfalls and how to avoid them',
                            'Expert tips for optimal results'
                        ]
                    },
                    {
                        'h2': 'Looking Forward: Trends and Future Outlook',
                        'bullets': [
                            'Emerging trends and developments to watch',
                            'Preparing for future changes and opportunities',
                            'Next steps and continued resources'
                        ]
                    }
                ]
            }
        }
        
        template = templates.get(article_type, templates['general'])
        
        if keyword:
            template['h1'] = f"{template['h1'].split(':')[0]}: {keyword.title()} Edition"
        
        template['ctas'] = [
            {"after": 0, "text": f"Want to master {topic}? Continue reading for expert insights."},
            {"after": 1, "text": "Ready to take action? Apply these proven strategies today."},
            {"after": 3, "text": "Start implementing these tips now to see real results!"}
        ]
        
        return template
    
    def format_outline_text(self, outline: Dict) -> str:
        """Format outline as plain text for download."""
        text = f"# {outline['h1']}\n\n"
        
        for idx, section in enumerate(outline['sections']):
            text += f"## {section['h2']}\n\n"
            for bullet in section['bullets']:
                text += f"• {bullet}\n"
            text += "\n"
            
            cta = next((c for c in outline['ctas'] if c['after'] == idx), None)
            if cta:
                text += f"💡 **Call to Action:** {cta['text']}\n\n"
        
        return text


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="AI Article Outline Generator",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
            margin-bottom: 2rem;
        }
        .outline-section {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid #4F46E5;
        }
        .cta-box {
            background-color: #E0F2FE;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid #0EA5E9;
        }
        .free-badge {
            background-color: #10B981;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
            margin-top: 0.5rem;
        }
        .article-type-badge {
            background-color: #4F46E5;
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 8px;
            font-size: 0.75rem;
            margin-left: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='main-header'>", unsafe_allow_html=True)
    st.title("📝 AI Article Outline Generator")
    st.markdown("Generate structured, SEO-optimized article outlines using **Hugging Face Serverless Inference API**")
    st.markdown("<span class='free-badge'>FREE - NO PAYMENT REQUIRED</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        default_token = os.environ.get("HF_TOKEN", "")
        
        if default_token:
            st.success("✅ Token loaded from environment")
            api_token = default_token
            st.session_state.api_token = api_token
        else:
            api_token = st.text_input(
                "🔑 Hugging Face Token",
                type="password",
                help="Get your FREE token from https://huggingface.co/settings/tokens"
            )
            
            if api_token:
                st.success("✅ Token provided")
                st.session_state.api_token = api_token
            else:
                st.warning("⚠️ Please enter your FREE token")
                st.markdown("[Get FREE token →](https://huggingface.co/settings/tokens)")
        
        st.markdown("---")
        st.success("**✨ Features:**")
        st.markdown("""
        ✅ How-to Guides  
        ✅ Listicles & Comparisons  
        ✅ Explanatory Articles  
        ✅ SEO-Optimized  
        ✅ CTA Suggestions  
        ✅ **Template Fallback (Always Works!)**  
        """)
        
        st.markdown("---")
        st.info("**Quick Start:**")
        st.markdown("""
        1. Get [free token](https://huggingface.co/settings/tokens)  
           *(Create token with 'read' permissions)*
        2. Enter article topic
        3. Add keyword (optional)
        4. Generate outline
        5. Download or copy
        
        **Note:** If AI models are unavailable, template generation ensures you always get results!
        """)
        
        st.markdown("---")
        st.markdown("**Version:** 2.2 (Free Tier)")
        st.markdown("**Always Works:** Template Fallback")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        topic = st.text_input(
            "📌 Article Topic *",
            placeholder="e.g., how to buy a phone",
            help="Enter your article topic - the AI will detect the type automatically"
        )
    
    with col2:
        keyword = st.text_input(
            "🔑 Target Keyword (optional)",
            placeholder="e.g., phone buying guide 2024",
            help="Enter your target SEO keyword for better optimization"
        )
    
    if topic.strip():
        generator_temp = HuggingFaceOutlineGenerator()
        detected_type = generator_temp.detect_article_type(topic)
        type_labels = {
            'how_to': 'How-To Guide',
            'listicle': 'Listicle/Comparison',
            'explanatory': 'Explanatory',
            'general': 'General Article'
        }
        st.markdown(f"**Detected Type:** <span class='article-type-badge'>{type_labels.get(detected_type, 'General')}</span>", unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        generate_button = st.button(
            "✨ Generate Outline",
            type="primary",
            use_container_width=True,
            disabled=not (topic.strip() and api_token)
        )
    
    if generate_button and topic.strip() and api_token:
        with st.spinner("🤖 Generating detailed outline..."):
            try:
                if 'generator' not in st.session_state or st.session_state.get('api_token') != api_token:
                    st.session_state.generator = HuggingFaceOutlineGenerator(api_token=api_token)
                    st.session_state.api_token = api_token
                
                outline = st.session_state.generator.generate_outline(topic, keyword)
                st.session_state.outline = outline
                st.success("✅ Outline generated successfully!")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("💡 Don't worry! Using template-based generation instead...")
                # Force template generation
                generator = HuggingFaceOutlineGenerator(api_token=api_token)
                outline = generator._generate_template_based(topic, keyword)
                st.session_state.outline = outline
                st.session_state.generator = generator
                st.success("✅ Outline generated using template!")
    
    if 'outline' in st.session_state:
        outline = st.session_state.outline
        
        st.markdown("---")
        st.header("📄 Generated Outline")
        
        col_act1, col_act2, col_act3 = st.columns([3, 1, 1])
        
        with col_act2:
            outline_text = st.session_state.generator.format_outline_text(outline)
            st.download_button(
                label="⬇️ Download TXT",
                data=outline_text,
                file_name=f"{re.sub(r'[^a-z0-9]', '-', topic.lower()[:50])}-outline.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col_act3:
            if st.button("📋 Show Copy Text", use_container_width=True):
                st.code(outline_text, language=None)
                st.info("👆 Select and copy the text above")
        
        st.markdown(f"### 🎯 Main Heading (H1)")
        st.markdown(f"<div class='outline-section'><h2>{outline['h1']}</h2></div>", unsafe_allow_html=True)
        
        for idx, section in enumerate(outline['sections']):
            st.markdown(f"### 📑 Section {idx + 1} (H2)")
            st.markdown(f"<div class='outline-section'>", unsafe_allow_html=True)
            st.markdown(f"**{section['h2']}**")
            
            for bullet in section['bullets']:
                st.markdown(f"• {bullet}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            cta = next((c for c in outline['ctas'] if c['after'] == idx), None)
            if cta:
                st.markdown(f"<div class='cta-box'>", unsafe_allow_html=True)
                st.markdown("**📢 Call to Action**")
                st.markdown(f"*{cta['text']}*")
                st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        if api_token:
            st.info("""
            👋 **Ready to generate!** Enter your article topic and click "Generate Outline".
            
            **Supported Article Types:**
            - 📖 How-to Guides & Tutorials
            - 📊 Listicles & Comparisons
            - 📚 Explanatory Articles
            - 📝 General Content
            
            The AI will automatically detect your article type and create an optimized outline!
            
            **Note:** If AI models are unavailable, template-based generation will provide excellent results!
            """)
        else:
            st.warning("""
            🔑 **FREE Hugging Face Token Required**
            
            1. Visit [Hugging Face Settings](https://huggingface.co/settings/tokens)
            2. Sign up for FREE (no credit card needed)
            3. Click "New token"
            4. Select **Read** permissions
            5. Copy the FULL token (starts with hf_)
            6. Paste token in sidebar
            
            **100% FREE - No payment required!**
            """)
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.875rem;'>
            <p>Powered by HuggingFace + Smart Template Fallback</p>
            <p>100% Free & Open Source | v2.2 - Always Works!</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
