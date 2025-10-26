"""
AI Article Outline Generator for Hugging Face Spaces
Fixed version with Inference Providers API compatibility
"""

import streamlit as st
import json
import re
import os
from typing import Dict, Optional
from huggingface_hub import InferenceClient
import time


class HuggingFaceOutlineGenerator:
    """Generate article outlines using Hugging Face Inference Providers."""
    
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.environ.get("HF_TOKEN")
        self.model_name = "meta-llama/Llama-3.2-3B-Instruct"
        self.client = None
        
    def initialize_client(self):
        """Initialize the Hugging Face client with error handling."""
        if not self.api_token:
            raise ValueError("Hugging Face token required. Get it from https://huggingface.co/settings/tokens")
    
        try:
            self.client = InferenceClient(api_key=self.api_token)
        except Exception as e:
            st.error(f"Failed to initialize client: {str(e)}")
            raise

    
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
        """Generate article outline using Hugging Face Inference Providers with retry logic."""
        if self.client is None:
            self.initialize_client()
    
        prompt = self.create_prompt(topic, keyword)
    
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Use NEW Inference Providers chat.completions API
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=1500,
                    temperature=0.7
                )
                
                # Extract response from new API structure
                response_text = completion.choices[0].message.content
                outline = self._parse_response(response_text, topic, keyword)
                return outline
            
            except Exception as e:
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "429" in error_msg:
                    if attempt < max_retries - 1:
                        st.warning(f"⏳ Rate limit hit, retrying in 3 seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(3)
                        continue
                    else:
                        st.error("⏳ Rate limit reached. Using template fallback.")
                elif "token" in error_msg or "401" in error_msg or "unauthorized" in error_msg:
                    st.error("🔑 Invalid or expired token. Please check your Hugging Face token.")
                    st.info("Get a new token at: https://huggingface.co/settings/tokens")
                elif "404" in error_msg or "not found" in error_msg:
                    st.error("❌ Model not available. Using template fallback.")
                else:
                    st.warning(f"API error: {str(e)}")
            
                return self._generate_template_based(topic, keyword)
    
        return self._generate_template_based(topic, keyword)

    
    def _parse_response(self, response: str, topic: str, keyword: str) -> Dict:
        """Parse API response and extract JSON outline."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                outline = json.loads(json_str)
                
                if self._validate_outline(outline):
                    return outline
            
            # If parsing fails, use template
            return self._generate_template_based(topic, keyword)
            
        except json.JSONDecodeError:
            return self._generate_template_based(topic, keyword)
    
    def _validate_outline(self, outline: Dict) -> bool:
        """Validate outline structure."""
        required_keys = ['h1', 'sections', 'ctas']
        if not all(key in outline for key in required_keys):
            return False
        
        if not isinstance(outline['sections'], list) or len(outline['sections']) < 3:
            return False
        
        for section in outline['sections']:
            if not all(key in section for key in ['h2', 'bullets']):
                return False
            if not isinstance(section['bullets'], list) or len(section['bullets']) < 2:
                return False
        
        return True
    
    def _generate_template_based(self, topic: str, keyword: str) -> Dict:
        """Enhanced template-based outline generation."""
        article_type = self.detect_article_type(topic)
        
        outline = {'h1': '', 'sections': [], 'ctas': []}
        
        if article_type == 'how_to':
            outline = self._generate_howto_outline(topic, keyword)
        elif article_type == 'listicle':
            outline = self._generate_listicle_outline(topic, keyword)
        elif article_type == 'explanatory':
            outline = self._generate_explanatory_outline(topic, keyword)
        else:
            outline = self._generate_general_outline(topic, keyword)
        
        return outline
    
    def _generate_howto_outline(self, topic: str, keyword: str) -> Dict:
        """Generate how-to guide outline."""
        h1 = f"{topic}" + (f" - {keyword}" if keyword else "")
        
        return {
            'h1': h1,
            'sections': [
                {
                    'h2': 'What You Need to Know Before Starting',
                    'bullets': [
                        'Essential background information and prerequisites',
                        'Tools and resources you will need',
                        'Time commitment and difficulty level'
                    ]
                },
                {
                    'h2': 'Step-by-Step Process',
                    'bullets': [
                        'Detailed first step with clear instructions',
                        'Second step building on the first',
                        'Third step to complete the process'
                    ]
                },
                {
                    'h2': 'Common Mistakes to Avoid',
                    'bullets': [
                        'First common pitfall and how to prevent it',
                        'Second mistake beginners often make',
                        'Third issue and its solution'
                    ]
                },
                {
                    'h2': 'Tips for Success and Next Steps',
                    'bullets': [
                        'Pro tip to optimize your results',
                        'How to troubleshoot common issues',
                        'Advanced techniques and further learning'
                    ]
                }
            ],
            'ctas': [
                {'after': 0, 'text': 'Ready to get started? Download our comprehensive checklist'},
                {'after': 2, 'text': 'Need help? Join our community for support'},
                {'after': 3, 'text': 'Master this skill - enroll in our advanced course'}
            ]
        }
    
    def _generate_listicle_outline(self, topic: str, keyword: str) -> Dict:
        """Generate listicle/comparison outline."""
        h1 = f"{topic}" + (f" - {keyword}" if keyword else "")
        
        return {
            'h1': h1,
            'sections': [
                {
                    'h2': 'Selection Criteria and Methodology',
                    'bullets': [
                        'How we evaluated and ranked options',
                        'Key factors we considered in our analysis',
                        'Why these criteria matter for your decision'
                    ]
                },
                {
                    'h2': 'Top Choices - Detailed Comparison',
                    'bullets': [
                        'First option: strengths, weaknesses, and best use cases',
                        'Second option: unique features and value proposition',
                        'Third option: why it stands out from competitors'
                    ]
                },
                {
                    'h2': 'Feature-by-Feature Analysis',
                    'bullets': [
                        'Performance comparison across key metrics',
                        'Price and value assessment',
                        'User experience and ease of use'
                    ]
                },
                {
                    'h2': 'Final Verdict and Recommendations',
                    'bullets': [
                        'Best overall choice and why',
                        'Best option for specific needs or budgets',
                        'What to consider before making your decision'
                    ]
                }
            ],
            'ctas': [
                {'after': 0, 'text': 'Get our detailed comparison chart'},
                {'after': 2, 'text': 'See our top pick in action - watch the demo'},
                {'after': 3, 'text': 'Make your choice with confidence - read user reviews'}
            ]
        }
    
    def _generate_explanatory_outline(self, topic: str, keyword: str) -> Dict:
        """Generate explanatory article outline."""
        h1 = f"Understanding {topic}" + (f" - {keyword}" if keyword else "")
        
        return {
            'h1': h1,
            'sections': [
                {
                    'h2': 'What Is It? Core Definition',
                    'bullets': [
                        'Clear, simple explanation of the concept',
                        'Historical context and origin',
                        'Why it matters in today\'s context'
                    ]
                },
                {
                    'h2': 'How It Works - Key Mechanisms',
                    'bullets': [
                        'Fundamental principles explained simply',
                        'Step-by-step breakdown of the process',
                        'Real-world analogy to aid understanding'
                    ]
                },
                {
                    'h2': 'Practical Applications and Examples',
                    'bullets': [
                        'Common use cases in everyday life',
                        'Industry-specific applications',
                        'Case study demonstrating the concept'
                    ]
                },
                {
                    'h2': 'Common Questions and Misconceptions',
                    'bullets': [
                        'Most frequently asked questions answered',
                        'Myths debunked with facts',
                        'What beginners should know'
                    ]
                }
            ],
            'ctas': [
                {'after': 1, 'text': 'Want to dive deeper? Download our comprehensive guide'},
                {'after': 2, 'text': 'See it in action - explore our interactive examples'},
                {'after': 3, 'text': 'Still have questions? Connect with our expert community'}
            ]
        }
    
    def _generate_general_outline(self, topic: str, keyword: str) -> Dict:
        """Generate general article outline."""
        h1 = f"Complete Guide to {topic}" + (f" - {keyword}" if keyword else "")
        
        return {
            'h1': h1,
            'sections': [
                {
                    'h2': 'Introduction and Overview',
                    'bullets': [
                        'What this topic covers and why it matters',
                        'Who should read this and what you\'ll learn',
                        'Key takeaways and benefits'
                    ]
                },
                {
                    'h2': 'Core Concepts and Fundamentals',
                    'bullets': [
                        'Essential information you need to know',
                        'Important terminology explained',
                        'Foundation for understanding advanced topics'
                    ]
                },
                {
                    'h2': 'Advanced Insights and Analysis',
                    'bullets': [
                        'Deeper dive into complex aspects',
                        'Expert perspectives and best practices',
                        'Latest trends and developments'
                    ]
                },
                {
                    'h2': 'Practical Tips and Next Steps',
                    'bullets': [
                        'How to apply this knowledge effectively',
                        'Common challenges and solutions',
                        'Resources for further learning'
                    ]
                }
            ],
            'ctas': [
                {'after': 1, 'text': 'Get started with our beginner-friendly guide'},
                {'after': 2, 'text': 'Level up your knowledge with our expert resources'},
                {'after': 3, 'text': 'Join our community to continue learning'}
            ]
        }
    
    def format_outline_text(self, outline: Dict) -> str:
        """Format outline as plain text for download."""
        text = f"# {outline['h1']}\n\n"
        
        for idx, section in enumerate(outline['sections']):
            text += f"## {section['h2']}\n\n"
            for bullet in section['bullets']:
                text += f"- {bullet}\n"
            text += "\n"
            
            cta = next((c for c in outline['ctas'] if c['after'] == idx), None)
            if cta:
                text += f"**Call to Action:** {cta['text']}\n\n"
        
        return text


def main():
    st.set_page_config(
        page_title="AI Article Outline Generator",
        page_icon="📝",
        layout="wide"
    )
    
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 2rem 0;
        }
        .outline-section {
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .cta-box {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 5px;
        }
        .free-badge {
            background-color: #10b981;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 12px;
            font-size: 0.875rem;
            font-weight: bold;
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
    st.markdown("Generate structured, SEO-optimized article outlines using **Hugging Face Inference Providers**")
    st.markdown("<span class='free-badge'>FREE CREDITS INCLUDED</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Check for token in environment first (from Hugging Face Spaces secrets)
        default_token = os.environ.get("HF_TOKEN", "")
        token_source = "environment" if default_token else "manual"
        
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
                st.markdown("[Get FREE token](https://huggingface.co/settings/tokens)")
        
        st.markdown("---")
        st.success("**✨ Features:**")
        st.markdown("""
        ✅ How-to Guides  
        ✅ Listicles & Comparisons  
        ✅ Explanatory Articles  
        ✅ SEO-Optimized  
        ✅ CTA Suggestions  
        """)
        
        st.markdown("---")
        st.info("**Quick Start:**")
        st.markdown("""
        1. Get [free token](https://huggingface.co/settings/tokens)
        2. Enter article topic
        3. Add keyword (optional)
        4. Generate outline
        5. Download or copy
        """)
        
        st.markdown("---")
        st.markdown("**Model:** Llama-3.2-3B-Instruct")
        st.markdown("**API:** Inference Providers")
        st.markdown("**Hosting:** Hugging Face Spaces")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        topic = st.text_input(
            "📌 Article Topic *",
            placeholder="e.g., How to Start Email Marketing for Small Businesses",
            help="Enter your article topic - the AI will detect the type automatically"
        )
    
    with col2:
        keyword = st.text_input(
            "🔑 Target Keyword (optional)",
            placeholder="e.g., email marketing tips 2024",
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
        with st.spinner("🤖 Generating detailed outline... This may take 15-30 seconds..."):
            try:
                if 'generator' not in st.session_state or st.session_state.get('api_token') != api_token:
                    st.session_state.generator = HuggingFaceOutlineGenerator(api_token=api_token)
                    st.session_state.api_token = api_token
                
                outline = st.session_state.generator.generate_outline(topic, keyword)
                st.session_state.outline = outline
                st.success("✅ Outline generated successfully!")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("💡 Tip: Make sure your Hugging Face token is valid and has proper permissions")
    
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
            """)
        else:
            st.warning("""
            🔑 **FREE Hugging Face Token Required**
            
            1. Visit [Hugging Face](https://huggingface.co/settings/tokens)
            2. Sign up for FREE (no credit card)
            3. Create new access token
            4. Paste in sidebar
            
            **100% FREE - No payment ever required!**
            """)
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.875rem;'>
            <p>Powered by Llama-3.2-3B-Instruct via Inference Providers | Running on Hugging Face Spaces</p>
            <p>100% Free & Open Source</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()