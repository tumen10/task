import torch
from transformers import ParlerTTSMiniJenny30HProcessor, ParlerTTSMiniJenny30HForConditionalGeneration
from scipy.io.wavfile import write
import numpy as np

# Загрузка модели и процессора
processor = ParlerTTSMiniJenny30HProcessor.from_pretrained("parler-tts/parler-tts-mini-jenny-30H")
model = ParlerTTSMiniJenny30HForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-jenny-30H")

# Функция для генерации аудио
def generate_audio(text, output_file):
    inputs = processor(text, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(inputs.input_ids)
    audio = processor.batch_decode(output, skip_special_tokens=True)[0]
    
    # Сохранение аудио в файл
    audio = np.array(audio)  # Преобразуем в numpy array
    write(output_file, 22050, audio)

# Пример использования функции
generate_audio("Привет, это тест модели Parler TTS.", "output.wav")
