# ... (your imports and classes remain the same) ...

# --------------------------
# Load Model Function (ONLY ONE!)
# --------------------------

def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load checkpoint
        model_path = "best_model.pth"
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # DEBUG: Check if vocabularies loaded correctly
        print("Source vocab size:", len(checkpoint['src_bpe_vocab']))
        print("Target vocab size:", len(checkpoint['trg_bpe_vocab']))
        
        # Test if we can access some tokens
        if '<s>' in checkpoint['trg_bpe_vocab']:
            print("<s> token ID:", checkpoint['trg_bpe_vocab']['<s>'])
        if 'a' in checkpoint['trg_bpe_vocab']:
            print("'a' token ID:", checkpoint['trg_bpe_vocab']['a'])
        
        # Reconstruct tokenizers
        src_bpe = BPETokenizer()
        src_bpe.vocab = checkpoint['src_bpe_vocab']
        src_bpe.inverse = {i: tok for tok, i in src_bpe.vocab.items()}
        
        trg_bpe = BPETokenizer()
        trg_bpe.vocab = checkpoint['trg_bpe_vocab'] 
        trg_bpe.inverse = {i: tok for tok, i in trg_bpe.vocab.items()}
        
        # Test tokenization
        test_text = "جرم ہے تیری گلی سے سر جھکا کر لوٹنا"
        test_ids = src_bpe.encode(test_text)
        print("Test encoding length:", len(test_ids))
        print("First 10 tokens:", test_ids[:10])
        
        # Recreate model
        attention = Attention(config.hidden_dim, config.hidden_dim)
        enc = Encoder(len(src_bpe.vocab), config.embedding_dim, config.hidden_dim, 
                     config.encoder_layers, config.dropout, config.bidirectional)
        dec = DecoderWithAttention(len(trg_bpe.vocab), config.embedding_dim, config.hidden_dim, 
                                  config.hidden_dim, config.dropout, attention)
        model = Seq2Seq(enc, dec, src_bpe.vocab['<pad>'], trg_bpe.vocab['<s>'], device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        print("✅ Model loaded successfully!")
        return model, src_bpe, trg_bpe, device
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None, None, None, None

# --------------------------
# Translation Functions
# --------------------------
def greedy_decode(model, src, src_lens, trg_bpe, max_len=50):
    model.eval()
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src, src_lens)
        hidden = hidden[-1].unsqueeze(0)
        
        input = torch.tensor([trg_bpe.vocab['<s>']], device=model.device)
        decoded_ids = [trg_bpe.vocab['<s>']]
        
        for _ in range(max_len):
            output, hidden, _ = model.decoder(input, hidden, encoder_outputs)
            top1 = output.argmax(1).item()
            decoded_ids.append(top1)
            
            if top1 == trg_bpe.vocab['</s>']:
                break
                
            input = torch.tensor([top1], device=model.device)
        
        return decoded_ids

def translate_text(model, src_bpe, trg_bpe, urdu_text, device):
    """Translate a single Urdu sentence to Roman Urdu"""
    try:
        if not urdu_text.strip():
            return "Please enter some Urdu text"
            
        # Encode the input
        src_ids = src_bpe.encode(urdu_text)
        src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(1)
        src_lens = torch.tensor([len(src_ids)], device=device)
        
        # Translate
        pred_ids = greedy_decode(model, src_tensor, src_lens, trg_bpe)
        pred_text = trg_bpe.decode(pred_ids)
        
        return pred_text
        
    except Exception as e:
        return f"Translation error: {str(e)}"

# --------------------------
# Test Function (ADD THIS HERE)
# --------------------------
def test_model_translation(model, src_bpe, trg_bpe, device):
    """Test with the same example that worked in Colab"""
    test_text = "جرم ہے تیری گلی سے سر جھکا کر لوٹنا"
    
    try:
        with torch.no_grad():
            src_ids = src_bpe.encode(test_text)
            src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(1)
            src_lens = torch.tensor([len(src_ids)], device=device)
            
            pred_ids = greedy_decode(model, src_tensor, src_lens, trg_bpe)
            pred_text = trg_bpe.decode(pred_ids)
            
        print(f"Test input: {test_text}")
        print(f"Test output: {pred_text}")
        return pred_text
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return None

# --------------------------
# Streamlit App
# --------------------------
def main():
    # THIS MUST BE THE FIRST STREAMLIT COMMAND
    st.set_page_config(
        page_title="Urdu to Roman Urdu Translator",
        page_icon="🌐",
        layout="wide"
    )
    
    # Then the rest of your code
    st.title("🌐 Urdu to Roman Urdu Translator")
    st.markdown("""
    This AI-powered tool translates Urdu text to Roman Urdu using a neural translation model.
    Enter your Urdu text below and click 'Translate'!
    """)
    
    # Load model
    model, src_bpe, trg_bpe, device = load_model()

    
    # ======== ADD TEST CALL HERE ========
    if model is not None:
        # Run test translation
        test_result = test_model_translation(model, src_bpe, trg_bpe, device)
        if test_result:
            st.info(f"Model test: {test_result}")
        else:
            st.error("Model test failed!")
    # ======== END TEST CALL ========
    
    if model is None:
        return
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        urdu_input = st.text_area(
            "📝 Enter Urdu Text:",
            "آپ کا نام کیا ہے؟",
            height=150,
            help="Type or paste Urdu text here"
        )
        
        translate_btn = st.button("🚀 Translate", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### ℹ️ Examples:")
        examples = {
            "What is your name?": "آپ کا نام کیا ہے؟",
            "How are you?": "آپ کیسے ہیں؟",
            "Beautiful day": "یہ ایک خوبصورت دن ہے",
            "I'm learning": "میں Urdu سیکھ رہا ہوں"
        }
        
        for eng, urdu in examples.items():
            if st.button(f"{eng}: {urdu}", use_container_width=True):
                urdu_input = urdu
    
    # Translation result
    if translate_btn and urdu_input.strip():
        with st.spinner("🔄 Translating..."):
            translation = translate_text(model, src_bpe, trg_bpe, urdu_input.strip(), device)
        
        if translation and not translation.startswith("Error"):
            st.success("### 📖 Translation Result:")
            st.code(translation, language="text")
            
            # Additional info
            with st.expander("📊 Translation Details"):
                st.write(f"**Input:** {urdu_input}")
                st.write(f"**Input length:** {len(urdu_input)} characters")
                st.write(f"**Output length:** {len(translation)} characters")
                
        else:
            st.error(translation)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using PyTorch & Streamlit | BLEU: 0.30</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


