from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

# 加載 Whisper 模型和處理器
model_name = "openai/whisper-medium"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# 加載台語音訊文件
audio_file = "taiwanese_audio.wav"  # 替換為您的台語音訊文件
waveform, sample_rate = torchaudio.load(audio_file)

# Whisper 要求 16000Hz，需進行重採樣
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
waveform = transform(waveform)

# 處理音訊並生成輸出
inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
logits = model.generate(inputs.input_features)

# 獲取台語語音的轉錄結果
transcription = processor.batch_decode(logits, skip_special_tokens=True)[0]
print("台語輸入結果：", transcription)
