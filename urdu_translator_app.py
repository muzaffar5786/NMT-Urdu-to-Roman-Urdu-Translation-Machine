import streamlit as st
import torch
import torch.nn as nn
import re
from collections import defaultdict
from typing import List
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
import random

# Download NLTK data
nltk.download('punkt')

# --------------------------
# Configuration
# --------------------------
class Config:
    embedding_dim = 128
    hidden_dim = 256
    encoder_layers = 1
    decoder_layers = 1
    bidirectional = False
    dropout = 0.3

config = Config()

# --------------------------
# BPE Tokenizer
# --------------------------
class BPETokenizer:
    def __init__(self, vocab_size=2000, special_tokens=None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ['<pad>', '<unk>', '<s>', '</s>']
        self.vocab = {tok:i for i,tok in enumerate(self.special_tokens)}
        self.inverse = {i:tok for tok,i in self.vocab.items()}
        self.merges = []

    def _preprocess(self, text):
        text = re.sub(r'([\.\,\;\:\!\?\(\)\-"])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text.strip())
        return text

    def encode(self, text):
        text = self._preprocess(text)
        toks = [self.vocab['<s>']]
        for w in text.split():
            tokens = list(w) + ['</w>']
            for token in tokens:
                toks.append(self.vocab.get(token, self.vocab['<unk>']))
        toks.append(self.vocab['</s>'])
        return toks

    def decode(self, ids: List[int]):
        toks = []
        for i in ids:
            if i in self.inverse:
                t = self.inverse[i]
                if t not in self.special_tokens:
                    toks.append(t)
        s = ''.join(toks).replace('</w>', ' ')
        s = re.sub(r'\s+([\.\,\;\:\!\?\(\)\-"])', r'\1', s)
        s = re.sub(r'\s+', ' ', s.strip())
        return s

# --------------------------
# Model Architecture
# --------------------------
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        
        return torch.softmax(attention, dim=1)

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, 
                           dropout=dropout if n_layers>1 else 0, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_lengths):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths.cpu(), enforce_sorted=False)
        packed_out, (hidden, cell) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out)
        return outputs, hidden, cell

class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.vocab_size = vocab_size
        self.attention = attention
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim + enc_hid_dim, dec_hid_dim)
        self.fc_out = nn.Linear(emb_dim + enc_hid_dim + dec_hid_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        
        attn_weights = self.attention(hidden.squeeze(0), encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        context = torch.bmm(attn_weights, encoder_outputs)
        context = context.permute(1, 0, 2)
        
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        
        output = output.squeeze(0)
        context = context.squeeze(0)
        embedded = embedded.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))
        return prediction, hidden, attn_weights.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_sos_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        
    def forward(self, src, src_lengths, trg=None, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        max_trg_len = trg.shape[0] if trg is not None else 60
        
        outputs = torch.zeros(max_trg_len, batch_size, self.decoder.vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        hidden = hidden[-1].unsqueeze(0)
        
        input = torch.full((batch_size,), self.trg_sos_idx, dtype=torch.long, device=self.device)
        
        for t in range(max_trg_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            top1 = output.argmax(1)
            
            if trg is not None and random.random() < teacher_forcing_ratio:
                input = trg[t]
            else:
                input = top1
                
        return outputs

# --------------------------
# Load Model Function
# --------------------------
@st.cache_resource
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
# Test Function
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
    
    st.title("🌐 Urdu to Roman Urdu Translator")
    st.markdown("""
    This AI-powered tool translates Urdu text to Roman Urdu using a neural translation model.
    Enter your Urdu text below and click 'Translate'!
    """)
    
    # Load model
    model, src_bpe, trg_bpe, device = load_model()
    
    # Test model
    if model is not None:
        test_result = test_model_translation(model, src_bpe, trg_bpe, device)
        if test_result:
            st.info(f"Model test: {test_result}")
        else:
            st.error("Model test failed!")
    
    if model is None:
        st.error("Failed to load model. Please check if 'best_model.pth' exists.")
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
            if st.button(f"{eng}", use_container_width=True):
                urdu_input = urdu
    
    # Translation result
    if translate_btn and urdu_input.strip():
        with st.spinner("🔄 Translating..."):
            translation = translate_text(model, src_bpe, trg_bpe, urdu_input.strip(), device)
        
        if translation and not translation.startswith("Error"):
            st.success("### 📖 Translation Result:")
            st.code(translation, language="text")
            
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
        <p>Built with ❤️ using PyTorch & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
