from scipy.io import wavfile
from pesq import pesq

def evaluate_pesq(reference_file, synth_file):
    ref_rate, ref_audio = wavfile.read(reference_file)
    synth_rate, synth_audio = wavfile.read(synth_file)
    return pesq(ref_rate, ref_audio, synth_audio, 'wb')

# Пример оценки
pesq_score = evaluate_pesq("reference.wav", "output.wav")
print(f"PESQ Score: {pesq_score}")
