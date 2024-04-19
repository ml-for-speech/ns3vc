import torch
import librosa
import soundfile as sf
import gradio as gr
import os
from huggingface_hub import hf_hub_download
import numpy as np
from pydub import AudioSegment
from ns3vc.amphion.models.ns3_codec import (
    FACodecEncoder,
    FACodecDecoder,
    FACodecRedecoder,
    FACodecEncoderV2,
    FACodecDecoderV2,
)

fa_encoder = FACodecEncoder(
    ngf=32,
    up_ratios=[2, 4, 5, 5],
    out_channels=256,
)

fa_decoder = FACodecDecoder(
    in_channels=256,
    upsample_initial_channel=1024,
    ngf=32,
    up_ratios=[5, 5, 4, 2],
    vq_num_q_c=2,
    vq_num_q_p=1,
    vq_num_q_r=3,
    vq_dim=256,
    codebook_dim=8,
    codebook_size_prosody=10,
    codebook_size_content=10,
    codebook_size_residual=10,
    use_gr_x_timbre=True,
    use_gr_residual_f0=True,
    use_gr_residual_phone=True,
)

fa_redecoder = FACodecRedecoder()

fa_encoder_v2 = FACodecEncoderV2(
    ngf=32,
    up_ratios=[2, 4, 5, 5],
    out_channels=256,
)

fa_decoder_v2 = FACodecDecoderV2(
    in_channels=256,
    upsample_initial_channel=1024,
    ngf=32,
    up_ratios=[5, 5, 4, 2],
    vq_num_q_c=2,
    vq_num_q_p=1,
    vq_num_q_r=3,
    vq_dim=256,
    codebook_dim=8,
    codebook_size_prosody=10,
    codebook_size_content=10,
    codebook_size_residual=10,
    use_gr_x_timbre=True,
    use_gr_residual_f0=True,
    use_gr_residual_phone=True,
)





class NS3VC:
    def __init__(self, device='auto'):
        global encoder_ckpt, decoder_ckpt, redecoder_ckpt, encoder_v2_ckpt, decoder_v2_ckpt, fa_encoder, fa_decoder, fa_redecoder, fa_encoder_v2, fa_decoder_v2
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if type(device) == str:
            device = torch.device(device)
        self.device = device
        encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
        decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")
        redecoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_redecoder.bin")
        encoder_v2_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder_v2.bin")
        decoder_v2_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder_v2.bin")


        fa_encoder.load_state_dict(torch.load(encoder_ckpt))
        fa_decoder.load_state_dict(torch.load(decoder_ckpt))
        fa_redecoder.load_state_dict(torch.load(redecoder_ckpt))
        fa_encoder_v2.load_state_dict(torch.load(encoder_v2_ckpt))
        fa_decoder_v2.load_state_dict(torch.load(decoder_v2_ckpt))

        fa_encoder = fa_encoder.to(device)
        fa_decoder = fa_decoder.to(device)
        fa_redecoder = fa_redecoder.to(device)
        fa_encoder_v2 = fa_encoder_v2.to(device)
        fa_decoder_v2 = fa_decoder_v2.to(device)
        fa_encoder.eval()
        fa_decoder.eval()
        fa_redecoder.eval()
        fa_encoder_v2.eval()
        fa_decoder_v2.eval()
    def infer_file(self, input, sample, output):
        with torch.no_grad():
            wav_a, sr = librosa.load(input, sr=16000)
            wav_a = np.pad(wav_a, (0, 200 - len(wav_a) % 200))
            wav_a = torch.tensor(wav_a).to(self.device).unsqueeze(0).unsqueeze(0)
            wav_b, sr = librosa.load(sample, sr=16000)
            wav_b = np.pad(wav_b, (0, 200 - len(wav_b) % 200))
            wav_b = torch.tensor(wav_b).to(self.device).unsqueeze(0).unsqueeze(0)

            enc_out_a = fa_encoder_v2(wav_a)
            prosody_a = fa_encoder_v2.get_prosody_feature(wav_a)
            enc_out_b = fa_encoder_v2(wav_b)
            prosody_b = fa_encoder_v2.get_prosody_feature(wav_b)

            vq_post_emb_a, vq_id_a, _, quantized, spk_embs_a = fa_decoder_v2(
                enc_out_a, prosody_a, eval_vq=False, vq=True
            )
            vq_post_emb_b, vq_id_b, _, quantized, spk_embs_b = fa_decoder_v2(
                enc_out_b, prosody_b, eval_vq=False, vq=True
            )
            vq_post_emb_a_to_b = fa_decoder_v2.vq2emb(vq_id_a, use_residual=False)
            recon_wav_a_to_b = fa_decoder_v2.inference(vq_post_emb_a_to_b, spk_embs_b)

        sf.write(output, recon_wav_a_to_b[0, 0].cpu().numpy(), 16000)