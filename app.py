"""
AI Article Outline Generator for Hugging Face Spaces
Fixed version with improved error handling and compatibility
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
            self.client = InferenceClient(token=self.api_token)
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
        
        return f"""[INST] You are an expert content strategist and SEO specialist. Generate a detailed, structured article outline.

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
Respond with ONLY the JSON, no additional text. [/INST]"""
    
    def generate_outline(self, topic: str, keyword: str = "") -> Dict:
        """Generate article outline using Hugging Face Inference Providers with retry logic."""
        if self.client is None:
            self.initialize_client()
    
        prompt = self.create_prompt(topic, keyword)
    
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Use chat completion API instead of text_generation
                completion = self.client.chat_completion(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.7
            )
            
                response = completion.choices[0].message.content
                outline = self._parse_response(response, topic, keyword)
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
        clean_topic = re.sub(r'\bhow to\b', 'How to', topic, flags=re.IGNORECASE)
        keyword_phrase = f" - {keyword}" if keyword else ""
        
        return {
            'h1': f"Complete Guide: {clean_topic}{keyword_phrase}",
            'sections': [
                {
                    'h2': 'Getting Started: What You Need to Know',
                    'bullets': [
                        'Essential prerequisites and requirements before you begin',
                        'Common misconceptions debunked by experts',
                        'Tools, resources, and materials you\'ll need'
                    ]
                },
                {
                    'h2': 'Step-by-Step Process',
                    'bullets': [
                        'Detailed walkthrough of each step with clear instructions',
                        'Pro tips and best practices from experienced practitioners',
                        'Common mistakes to avoid and how to prevent them'
                    ]
                },
                {
                    'h2': 'Advanced Techniques and Optimization',
                    'bullets': [
                        'Level up your skills with advanced strategies',
                        'Expert shortcuts and time-saving techniques',
                        'Troubleshooting guide for common issues'
                    ]
                },
                {
                    'h2': 'Measuring Success and Next Steps',
                    'bullets': [
                        'Key performance indicators and metrics to track',
                        'How to evaluate your progress and results',
                        'Continuous improvement strategies and resources'
                    ]
                }
            ],
            'ctas': [
                {'after': 0, 'text': 'Download our free beginner\'s checklist and resource guide'},
                {'after': 1, 'text': 'Watch our step-by-step video tutorial (free access)'},
                {'after': 3, 'text': 'Schedule a free 15-minute consultation with our experts'}
            ]
        }
    
    def _generate_listicle_outline(self, topic: str, keyword: str) -> Dict:
        """Generate listicle/comparison outline."""
        keyword_phrase = f" | {keyword}" if keyword else ""
        main_heading = topic if any(word in topic for word in ['Best', 'Top', 'vs']) else f"Best {topic}"
        
        return {
            'h1': f"{main_heading} - Expert Review & Comparison{keyword_phrase}",
            'sections': [
                {
                    'h2': 'Our Testing Methodology and Selection Criteria',
                    'bullets': [
                        'Rigorous testing process: how we evaluated each option over 30+ hours',
                        'Key factors we considered: features, pricing, ease of use, and support',
                        'Why our recommendations are trustworthy and unbiased'
                    ]
                },
                {
                    'h2': 'Quick Comparison: Top Picks at a Glance',
                    'bullets': [
                        'Side-by-side feature comparison of all top options',
                        'Pricing breakdown: from budget-friendly to premium solutions',
                        'Best use cases: which option fits your specific needs'
                    ]
                },
                {
                    'h2': 'Detailed Reviews: Deep Dive into Each Option',
                    'bullets': [
                        'In-depth analysis of features, performance, and value',
                        'Comprehensive pros and cons based on real-world testing',
                        'Real user experiences, testimonials, and ratings'
                    ]
                },
                {
                    'h2': 'Decision Guide: Choosing the Right Option for You',
                    'bullets': [
                        'Critical questions to ask yourself before deciding',
                        'Step-by-step decision framework based on your needs',
                        'Implementation tips and getting started advice'
                    ]
                }
            ],
            'ctas': [
                {'after': 1, 'text': 'Download the complete comparison spreadsheet (free)'},
                {'after': 2, 'text': 'Take our 2-minute recommendation quiz'},
                {'after': 3, 'text': 'Get personalized recommendations from our team'}
            ]
        }
    
    def _generate_explanatory_outline(self, topic: str, keyword: str) -> Dict:
        """Generate explanatory article outline."""
        keyword_phrase = f" ({keyword})" if keyword else ""
        
        return {
            'h1': f"Understanding {topic}{keyword_phrase}: A Complete Guide",
            'sections': [
                {
                    'h2': 'Introduction: What You Need to Know',
                    'bullets': [
                        f'Clear definition and explanation of {topic}',
                        'Why this topic matters in today\'s context',
                        'Who should care about this and why it\'s relevant'
                    ]
                },
                {
                    'h2': 'Core Concepts and Fundamentals',
                    'bullets': [
                        'Essential principles explained in simple terms',
                        'Key terminology and definitions you need to know',
                        'Historical context and how it evolved over time'
                    ]
                },
                {
                    'h2': 'Real-World Applications and Examples',
                    'bullets': [
                        'Practical use cases and scenarios from various industries',
                        'Success stories and compelling case studies',
                        'Step-by-step implementation strategies'
                    ]
                },
                {
                    'h2': 'Future Outlook and Expert Insights',
                    'bullets': [
                        'Current trends and what industry experts are saying',
                        'Upcoming developments and innovations to watch',
                        'How to stay informed and ahead of the curve'
                    ]
                }
            ],
            'ctas': [
                {'after': 1, 'text': 'Subscribe to our newsletter for weekly insights'},
                {'after': 2, 'text': 'Download our free comprehensive resource guide'},
                {'after': 3, 'text': 'Join our community of 10,000+ enthusiasts'}
            ]
        }
    
    def _generate_general_outline(self, topic: str, keyword: str) -> Dict:
        """Generate general article outline."""
        keyword_phrase = f" - {keyword}" if keyword else ""
        
        return {
            'h1': f"{topic}: Everything You Need to Know{keyword_phrase}",
            'sections': [
                {
                    'h2': 'Introduction and Overview',
                    'bullets': [
                        f'Comprehensive introduction to {topic}',
                        'Current state and why this matters now',
                        'Who this guide is for and what you\'ll learn'
                    ]
                },
                {
                    'h2': 'Key Concepts and Important Details',
                    'bullets': [
                        'Core principles and fundamental concepts explained',
                        'Essential terminology and definitions',
                        'Historical background and evolution'
                    ]
                },
                {
                    'h2': 'Practical Applications and Use Cases',
                    'bullets': [
                        'Real-world applications across different contexts',
                        'Success stories and proven case studies',
                        'Implementation strategies and best practices'
                    ]
                },
                {
                    'h2': 'Future Trends and Action Steps',
                    'bullets': [
                        'Expert predictions and emerging trends',
                        'Upcoming developments to keep on your radar',
                        'Actionable next steps and resources'
                    ]
                }
            ],
            'ctas': [
                {'after': 1, 'text': 'Get our free beginner\'s guide delivered to your inbox'},
                {'after': 2, 'text': 'Download our complete resource library'},
                {'after': 3, 'text': 'Connect with our community of experts'}
            ]
        }
    
    def format_outline_text(self, outline: Dict) -> str:
        """Format outline as plain text for download/copy."""
        text = f"{outline['h1']}\n{'=' * len(outline['h1'])}\n\n"
        
        for idx, section in enumerate(outline['sections']):
            text += f"{section['h2']}\n{'-' * len(section['h2'])}\n"
            for bullet in section['bullets']:
                text += f"• {bullet}\n"
            
            cta = next((c for c in outline['ctas'] if c['after'] == idx), None)
            if cta:
                text += f"\n[CTA: {cta['text']}]\n"
            text += '\n'
        
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
        .main-header {text-align: center; padding: 2rem 0;}
        .outline-section {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 4px solid #4F46E5;
        }
        .cta-box {
            background-color: #D1FAE5;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #10B981;
            margin: 1rem 0;
        }
        .free-badge {
            background-color: #10B981;
            color: white;
            padding: 0.25rem 0.75rem;
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
