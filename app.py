"""
AI Article Outline Generator for Hugging Face Spaces
Enhanced version with multiple model fallback support
"""

import streamlit as st
import json
import re
import os
from typing import Dict, Optional, List, Tuple
from huggingface_hub import InferenceClient
import time


class HuggingFaceOutlineGenerator:
    """Generate article outlines using Hugging Face Inference Providers."""
    
    # List of models to try in order (from best to fallback)
    MODELS = [
        "meta-llama/Llama-3.2-3B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "HuggingFaceH4/zephyr-7b-beta",
        "google/gemma-2-2b-it"
    ]
    
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.environ.get("HF_TOKEN")
        self.model_name = None  # Will be set after testing
        self.client = None
        self.working_model_found = False
        
    def initialize_client(self):
        """Initialize the Hugging Face client with error handling."""
        if not self.api_token:
            raise ValueError("Hugging Face token required. Get it from https://huggingface.co/settings/tokens")
    
        try:
            self.client = InferenceClient(token=self.api_token)
            # Try to find a working model
            self._find_working_model()
        except Exception as e:
            st.error(f"Failed to initialize client: {str(e)}")
            raise
    
    def _find_working_model(self):
        """Test models to find one that works with Inference Providers."""
        if self.working_model_found:
            return
        
        for model in self.MODELS:
            try:
                # Quick test with minimal tokens
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5,
                    timeout=10
                )
                
                # If we get here, the model works!
                self.model_name = model
                self.working_model_found = True
                st.success(f"✅ Using model: {model.split('/')[-1]}")
                return
                
            except Exception as e:
                error_msg = str(e).lower()
                # Only continue if it's a model availability issue
                if "404" in error_msg or "not found" in error_msg or "not available" in error_msg:
                    continue
                else:
                    # Other errors (like auth) should be raised
                    if "token" in error_msg or "401" in error_msg:
                        raise
        
        # If no model worked
        st.warning("⚠️ Could not connect to any Inference Provider model. Using template fallback.")
        self.model_name = self.MODELS[0]  # Set a default for display purposes

    
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
        
        # If no working model was found during initialization, use template
        if not self.working_model_found:
            st.info("📋 Using template-based generation (AI models unavailable)")
            return self._generate_template_based(topic, keyword)
    
        prompt = self.create_prompt(topic, keyword)
    
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Use Inference Providers chat.completions API
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
                
                # Extract response from API structure
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
                    st.error("❌ Model temporarily unavailable. Using template fallback.")
                else:
                    st.warning(f"⚠️ API error: {str(e)[:100]}")
            
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
        clean_topic = topic.replace("how to", "").replace("How to", "").strip()
        kw = f" ({keyword})" if keyword else ""
        
        return {
            'h1': f"Complete Guide: How to {clean_topic.title()}{kw}",
            'sections': [
                {
                    'h2': f"Understanding {clean_topic.title()}: What You Need to Know",
                    'bullets': [
                        f"Core concepts and fundamentals of {clean_topic}",
                        f"Common misconceptions about {clean_topic}",
                        f"Who can benefit from {clean_topic} and why"
                    ]
                },
                {
                    'h2': f"Essential Preparation Before You Start",
                    'bullets': [
                        "Required tools, resources, and prerequisites",
                        "Setting realistic goals and expectations",
                        "Common mistakes to avoid from the beginning"
                    ]
                },
                {
                    'h2': f"Step-by-Step: How to {clean_topic.title()}",
                    'bullets': [
                        "Detailed walkthrough of each critical step",
                        "Pro tips and best practices for better results",
                        "Time-saving shortcuts and efficiency hacks"
                    ]
                },
                {
                    'h2': f"Troubleshooting and Advanced Tips",
                    'bullets': [
                        "Solutions to common problems and challenges",
                        "Advanced techniques for experienced users",
                        "Resources for continued learning and improvement"
                    ]
                }
            ],
            'ctas': [
                {'after': 0, 'text': f"Ready to master {clean_topic}? Continue reading for our proven step-by-step process."},
                {'after': 2, 'text': f"Download our free {clean_topic} checklist to track your progress."},
                {'after': 3, 'text': f"Need expert help? Join our community of {clean_topic} enthusiasts."}
            ]
        }
    
    def _generate_listicle_outline(self, topic: str, keyword: str) -> Dict:
        """Generate listicle/comparison outline."""
        kw = f" - {keyword}" if keyword else ""
        
        return {
            'h1': f"{topic.title()}: Expert Analysis and Comparison{kw}",
            'sections': [
                {
                    'h2': "Selection Criteria: What Makes a Great Choice",
                    'bullets': [
                        "Key factors to consider when evaluating options",
                        "How we tested and ranked each option",
                        "Understanding your specific needs and priorities"
                    ]
                },
                {
                    'h2': "Top Picks: Detailed Reviews and Analysis",
                    'bullets': [
                        "In-depth review of the best options available",
                        "Pros and cons of each recommendation",
                        "Real-world performance and user feedback"
                    ]
                },
                {
                    'h2': "Head-to-Head Comparison",
                    'bullets': [
                        "Side-by-side feature and pricing comparison",
                        "Which option is best for different use cases",
                        "Value for money analysis and recommendations"
                    ]
                },
                {
                    'h2': "Making Your Decision: Final Recommendations",
                    'bullets': [
                        "Our top recommendation for most users",
                        "Budget-friendly alternatives worth considering",
                        "Premium options for advanced users"
                    ]
                }
            ],
            'ctas': [
                {'after': 0, 'text': "See our methodology and testing process to understand our rankings."},
                {'after': 2, 'text': "Compare all options side-by-side with our interactive comparison tool."},
                {'after': 3, 'text': "Get our exclusive discount codes for the top-rated options."}
            ]
        }
    
    def _generate_explanatory_outline(self, topic: str, keyword: str) -> Dict:
        """Generate explanatory article outline."""
        clean_topic = topic.replace("what is", "").replace("What is", "").strip()
        kw = f": {keyword}" if keyword else ""
        
        return {
            'h1': f"Understanding {clean_topic.title()}{kw} - Complete Explanation",
            'sections': [
                {
                    'h2': f"What is {clean_topic.title()}? Definition and Overview",
                    'bullets': [
                        f"Clear, simple definition of {clean_topic}",
                        "Historical context and how it developed",
                        "Why it matters in today's context"
                    ]
                },
                {
                    'h2': f"How {clean_topic.title()} Works: Key Concepts",
                    'bullets': [
                        "Breaking down the fundamental principles",
                        "Real-world examples to illustrate the concept",
                        "Common terminology and what it means"
                    ]
                },
                {
                    'h2': f"Applications and Use Cases",
                    'bullets': [
                        f"Practical applications of {clean_topic}",
                        "Industries and fields where it's most relevant",
                        "Benefits and advantages of understanding this topic"
                    ]
                },
                {
                    'h2': f"Common Questions and Misconceptions",
                    'bullets': [
                        "Frequently asked questions answered",
                        "Debunking common myths and misconceptions",
                        "Expert insights and future trends"
                    ]
                }
            ],
            'ctas': [
                {'after': 0, 'text': f"Want to dive deeper? Explore our advanced guide to {clean_topic}."},
                {'after': 2, 'text': f"See how {clean_topic} applies to your situation with our interactive tool."},
                {'after': 3, 'text': f"Stay updated with the latest developments in {clean_topic}."}
            ]
        }
    
    def _generate_general_outline(self, topic: str, keyword: str) -> Dict:
        """Generate general article outline."""
        kw = f" - {keyword}" if keyword else ""
        
        return {
            'h1': f"{topic.title()}: Comprehensive Guide{kw}",
            'sections': [
                {
                    'h2': "Introduction and Background",
                    'bullets': [
                        f"Overview of {topic} and its significance",
                        "Current trends and latest developments",
                        "Who should read this guide and why"
                    ]
                },
                {
                    'h2': "Key Concepts and Fundamentals",
                    'bullets': [
                        "Essential information you need to know",
                        "Important terminology and definitions",
                        "Core principles explained clearly"
                    ]
                },
                {
                    'h2': "Practical Applications and Examples",
                    'bullets': [
                        "Real-world use cases and scenarios",
                        "Step-by-step guidance and best practices",
                        "Tips for getting the best results"
                    ]
                },
                {
                    'h2': "Conclusion and Next Steps",
                    'bullets': [
                        "Summary of key takeaways",
                        "Action items and recommendations",
                        "Additional resources for further learning"
                    ]
                }
            ],
            'ctas': [
                {'after': 0, 'text': "Continue reading to discover everything you need to know."},
                {'after': 2, 'text': "Download our free resource guide for easy reference."},
                {'after': 3, 'text': "Join our newsletter for more expert insights and tips."}
            ]
        }
    
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
                text += f"[CTA: {cta['text']}]\n\n"
        
        return text


def main():
    st.set_page_config(
        page_title="AI Article Outline Generator",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        .main-header {
            padding: 1rem 0;
            border-bottom: 2px solid #f0f2f6;
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
            background-color: #FEF3C7;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid #F59E0B;
        }
        .free-badge {
            background-color: #10B981;
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 8px;
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
        ✅ Auto Model Fallback  
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
        st.markdown("**Models:** Multiple with auto-fallback")
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
            <p>Powered by Multiple LLMs via Inference Providers | Running on Hugging Face Spaces</p>
            <p>100% Free & Open Source</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()