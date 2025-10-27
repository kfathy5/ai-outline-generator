"""
AI Article Outline Generator - Llama Version
Requires: Accepting Llama terms on HuggingFace first
"""

import streamlit as st
import json
import re
import os
from typing import Dict, Optional
from huggingface_hub import InferenceClient
import time


class HuggingFaceOutlineGenerator:
    """Generate article outlines using Llama model."""
    
    # Primary model: Llama (requires accepting terms on HF)
    # Fallbacks: Free-tier models
    MODELS = [
        "meta-llama/Llama-3.2-3B-Instruct",  # Primary (requires terms acceptance)
        "google/flan-t5-large",               # Fallback 1
        "google/flan-t5-base",                # Fallback 2
    ]
    
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.environ.get("HF_TOKEN")
        self.model_name = None
        self.client = None
        self.working_model_found = False
        
    def initialize_client(self):
        """Initialize the Hugging Face client."""
        if not self.api_token:
            raise ValueError("Hugging Face token required")
    
        try:
            self.client = InferenceClient(token=self.api_token)
            self._find_working_model()
        except Exception as e:
            st.error(f"Failed to initialize: {str(e)}")
            raise
    
    def _find_working_model(self):
        """Test models to find working one."""
        if self.working_model_found:
            return
        
        st.info("🔍 Testing models...")
        
        for model in self.MODELS:
            try:
                st.text(f"Testing {model.split('/')[-1]}...")
                
                # Test the model
                response = self.client.text_generation(
                    prompt="Hello",
                    model=model,
                    max_new_tokens=5
                )
                
                self.model_name = model
                self.working_model_found = True
                st.success(f"✅ Connected to: {model.split('/')[-1]}")
                
                # Show which model is being used
                if "llama" in model.lower():
                    st.info("🦙 Using Llama model - best quality!")
                elif "flan-t5" in model.lower():
                    st.info("📝 Using FLAN-T5 - good quality fallback")
                
                return
                
            except Exception as e:
                error_msg = str(e).lower()
                
                if "401" in error_msg or "unauthorized" in error_msg:
                    st.error("🔑 Token issue - please verify your HF token")
                    raise ValueError("Invalid token")
                elif "403" in error_msg or "gated" in error_msg or "access" in error_msg:
                    st.warning(f"⚠️ {model.split('/')[-1]}: Need to accept terms on HuggingFace")
                    st.info(f"Visit: https://huggingface.co/{model}")
                    st.info("Click 'Agree and access repository', then try again in 5-10 minutes")
                    continue
                elif "404" in error_msg:
                    st.warning(f"❌ {model.split('/')[-1]}: Not available")
                    continue
                else:
                    st.warning(f"⚠️ {model.split('/')[-1]}: {str(e)[:100]}")
                    continue
        
        # If no model worked
        st.warning("⚠️ No AI models available. Using template generation.")
        self.model_name = self.MODELS[0]

    def detect_article_type(self, topic: str) -> str:
        """Detect article type."""
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
        """Create prompt for AI."""
        article_type = self.detect_article_type(topic)
        
        type_guidance = {
            'how_to': "Create a practical how-to guide with clear steps.",
            'listicle': "Create a comparison or list article.",
            'explanatory': "Create an educational article with examples.",
            'general': "Create a comprehensive article."
        }
        
        return f"""Generate a detailed article outline in JSON format.

Topic: {topic}
Keyword: {keyword if keyword else "Not specified"}
Type: {type_guidance.get(article_type, type_guidance['general'])}

Return ONLY valid JSON:
{{
    "h1": "SEO-optimized main heading",
    "sections": [
        {{
            "h2": "First section heading",
            "bullets": ["Point 1", "Point 2", "Point 3"]
        }},
        {{
            "h2": "Second section heading",
            "bullets": ["Point 1", "Point 2", "Point 3"]
        }},
        {{
            "h2": "Third section heading",
            "bullets": ["Point 1", "Point 2", "Point 3"]
        }},
        {{
            "h2": "Fourth section heading",
            "bullets": ["Point 1", "Point 2", "Point 3"]
        }}
    ],
    "ctas": [
        {{"after": 0, "text": "CTA after first section"}},
        {{"after": 1, "text": "CTA after second section"}},
        {{"after": 3, "text": "Final CTA"}}
    ]
}}"""
    
    def generate_outline(self, topic: str, keyword: str = "") -> Dict:
        """Generate outline."""
        if self.client is None:
            self.initialize_client()
        
        if not self.working_model_found:
            st.info("📋 Using template generation")
            return self._generate_template_based(topic, keyword)
    
        prompt = self.create_prompt(topic, keyword)
    
        try:
            response_text = self.client.text_generation(
                prompt=prompt,
                model=self.model_name,
                max_new_tokens=1500,
                temperature=0.7,
                return_full_text=False
            )
            
            return self._parse_response(response_text, topic, keyword)
            
        except Exception as e:
            st.error(f"Generation error: {str(e)[:200]}")
            return self._generate_template_based(topic, keyword)
    
    def _parse_response(self, response_text: str, topic: str, keyword: str) -> Dict:
        """Parse AI response."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                outline = json.loads(json_match.group(0))
                if self._validate_outline(outline):
                    return outline
            return self._generate_template_based(topic, keyword)
        except:
            return self._generate_template_based(topic, keyword)
    
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
    
    def _generate_template_based(self, topic: str, keyword: str = "") -> Dict:
        """Generate using template."""
        article_type = self.detect_article_type(topic)
        
        if article_type == 'how_to':
            h1 = f"How to {topic.replace('how to', '').strip().title()}: Complete Guide"
            sections = [
                {
                    'h2': 'Getting Started: Prerequisites',
                    'bullets': [
                        'Understanding requirements and tools',
                        'Setting up for success',
                        'Avoiding common mistakes'
                    ]
                },
                {
                    'h2': 'Step-by-Step Process',
                    'bullets': [
                        'Following the proven methodology',
                        'Best practices for each phase',
                        'Troubleshooting common issues'
                    ]
                },
                {
                    'h2': 'Advanced Techniques',
                    'bullets': [
                        'Taking results to the next level',
                        'Professional strategies',
                        'Measuring and improving outcomes'
                    ]
                },
                {
                    'h2': 'Long-Term Success',
                    'bullets': [
                        'Ongoing maintenance',
                        'Scaling your approach',
                        'Continued learning resources'
                    ]
                }
            ]
        else:
            h1 = f"{topic.title()}: Essential Guide"
            sections = [
                {
                    'h2': 'Introduction and Overview',
                    'bullets': [
                        f'Understanding {topic}',
                        'Why this topic matters',
                        'What you will learn'
                    ]
                },
                {
                    'h2': 'Key Components',
                    'bullets': [
                        'Main aspects and features',
                        'How elements work together',
                        'Critical success factors'
                    ]
                },
                {
                    'h2': 'Best Practices',
                    'bullets': [
                        'Proven strategies',
                        'Common pitfalls to avoid',
                        'Expert tips'
                    ]
                },
                {
                    'h2': 'Next Steps',
                    'bullets': [
                        'Emerging trends',
                        'Future opportunities',
                        'Resources for learning'
                    ]
                }
            ]
        
        if keyword:
            h1 = f"{h1.split(':')[0]}: {keyword.title()}"
        
        return {
            'h1': h1,
            'sections': sections,
            'ctas': [
                {"after": 0, "text": f"Ready to master {topic}? Continue reading!"},
                {"after": 1, "text": "Apply these strategies today for best results."},
                {"after": 3, "text": "Start implementing now!"}
            ]
        }
    
    def format_outline_text(self, outline: Dict) -> str:
        """Format as text."""
        text = f"# {outline['h1']}\n\n"
        
        for idx, section in enumerate(outline['sections']):
            text += f"## {section['h2']}\n\n"
            for bullet in section['bullets']:
                text += f"• {bullet}\n"
            text += "\n"
            
            cta = next((c for c in outline['ctas'] if c['after'] == idx), None)
            if cta:
                text += f"💡 {cta['text']}\n\n"
        
        return text


def main():
    """Main app."""
    st.set_page_config(
        page_title="AI Article Outline Generator",
        page_icon="📝",
        layout="wide"
    )
    
    st.markdown("""<style>
        .main-header { text-align: center; padding: 1rem 0; }
        .outline-section { background: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #4F46E5; }
        .cta-box { background: #E0F2FE; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #0EA5E9; }
        .badge { background: #10B981; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; }
    </style>""", unsafe_allow_html=True)
    
    st.markdown("<div class='main-header'>", unsafe_allow_html=True)
    st.title("📝 AI Article Outline Generator")
    st.markdown("Using **Llama 3.2** (if terms accepted) or **FLAN-T5** fallback")
    st.markdown("<span class='badge'>FREE</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        default_token = os.environ.get("HF_TOKEN", "")
        
        if default_token:
            st.success("✅ Token from environment")
            api_token = default_token
        else:
            api_token = st.text_input(
                "🔑 HuggingFace Token",
                type="password",
                help="Get from https://huggingface.co/settings/tokens"
            )
            
            if api_token:
                st.success("✅ Token provided")
            else:
                st.warning("⚠️ Enter your token")
        
        st.markdown("---")
        st.info("**To Use Llama Model:**")
        st.markdown("""
        1. Visit [Llama model page](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
        2. Click "Agree and access"
        3. Wait 5-10 minutes
        4. Then generate outline
        
        Or just use FLAN-T5 fallback!
        """)
        
        st.markdown("---")
        st.markdown("**Version:** 2.3 (Llama)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        topic = st.text_input(
            "📌 Article Topic *",
            placeholder="e.g., how to buy a phone"
        )
    
    with col2:
        keyword = st.text_input(
            "🔑 Target Keyword",
            placeholder="e.g., phone guide 2024"
        )
    
    if topic.strip():
        generator_temp = HuggingFaceOutlineGenerator()
        detected_type = generator_temp.detect_article_type(topic)
        type_labels = {
            'how_to': 'How-To Guide',
            'listicle': 'Listicle',
            'explanatory': 'Explanatory',
            'general': 'General'
        }
        st.markdown(f"**Type:** {type_labels.get(detected_type, 'General')}")
    
    generate_button = st.button(
        "✨ Generate Outline",
        type="primary",
        disabled=not (topic.strip() and api_token)
    )
    
    if generate_button and topic.strip() and api_token:
        with st.spinner("🤖 Generating..."):
            try:
                if 'generator' not in st.session_state or st.session_state.get('api_token') != api_token:
                    st.session_state.generator = HuggingFaceOutlineGenerator(api_token=api_token)
                    st.session_state.api_token = api_token
                
                outline = st.session_state.generator.generate_outline(topic, keyword)
                st.session_state.outline = outline
                st.success("✅ Outline generated!")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                generator = HuggingFaceOutlineGenerator(api_token=api_token)
                outline = generator._generate_template_based(topic, keyword)
                st.session_state.outline = outline
                st.session_state.generator = generator
                st.success("✅ Using template!")
    
    if 'outline' in st.session_state:
        outline = st.session_state.outline
        
        st.markdown("---")
        st.header("📄 Generated Outline")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            outline_text = st.session_state.generator.format_outline_text(outline)
            st.download_button(
                "⬇️ Download",
                data=outline_text,
                file_name=f"{re.sub(r'[^a-z0-9]', '-', topic.lower()[:50])}-outline.txt",
                mime="text/plain"
            )
        
        st.markdown(f"### 🎯 {outline['h1']}")
        
        for idx, section in enumerate(outline['sections']):
            st.markdown(f"<div class='outline-section'>", unsafe_allow_html=True)
            st.markdown(f"**{section['h2']}**")
            for bullet in section['bullets']:
                st.markdown(f"• {bullet}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            cta = next((c for c in outline['ctas'] if c['after'] == idx), None)
            if cta:
                st.markdown(f"<div class='cta-box'>📢 {cta['text']}</div>", unsafe_allow_html=True)
    else:
        if api_token:
            st.info("👋 Ready! Enter topic and generate.")
        else:
            st.warning("🔑 Please add your HuggingFace token in the sidebar")


if __name__ == "__main__":
    main()
