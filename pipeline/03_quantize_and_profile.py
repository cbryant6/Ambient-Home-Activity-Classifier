"""
Stage 3: Quantize and Profile
Usage: python 03_quantize_and_profile.py --model_dir ./models --data_dir ./processed
"""
import os, argparse, json
import numpy as np
import librosa
import tensorflow as tf

TARGET_SR = 16000
CLIP_DURATION = 1.0
CLASS_NAMES = ["quiet","human_activity","appliance_mechanical","transient_impact","environmental_ambient"]

def load_dataset(data_dir, split):
    audio_clips, labels = [], []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(data_dir, split, cls_name)
        if not os.path.exists(cls_dir): continue
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.endswith('.wav'): continue
            audio, _ = librosa.load(os.path.join(cls_dir, fname), sr=TARGET_SR, mono=True)
            target_len = int(TARGET_SR * CLIP_DURATION)
            audio = np.pad(audio, (0, max(0, target_len-len(audio))))[:target_len]
            audio_clips.append(audio); labels.append(cls_idx)
    return np.array(audio_clips, dtype=np.float32), np.array(labels, dtype=np.int32)

def compute_mfccs(audio_clips, n_mfcc, hop_length, n_fft):
    fixed_frames = 1 + int(TARGET_SR * CLIP_DURATION) // hop_length
    features = []
    n_mels = 40

    mel_min = 2595.0 * np.log10(1.0 + 0 / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + 8000.0 / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_points = np.floor((n_fft + 1) * hz_points / TARGET_SR).astype(int)
    bin_points = np.clip(bin_points, 0, n_fft // 2)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        for k in range(bin_points[m], bin_points[m+1]+1):
            if bin_points[m+1] != bin_points[m]:
                filterbank[m][k] = (k - bin_points[m]) / (bin_points[m+1] - bin_points[m])
        for k in range(bin_points[m+1], bin_points[m+2]+1):
            if bin_points[m+2] != bin_points[m+1]:
                filterbank[m][k] = (bin_points[m+2] - k) / (bin_points[m+2] - bin_points[m+1])

    for audio in audio_clips:
        mfcc_frames = []
        for f in range(fixed_frames):
            start = f * hop_length
            frame = np.zeros(n_fft)
            end = min(start + n_fft, len(audio))
            frame[:end-start] = audio[start:end]
            window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft) / (n_fft - 1)))
            frame *= window
            fft_out = np.fft.rfft(frame)
            power = (fft_out.real**2 + fft_out.imag**2) / n_fft
            mel_energy = filterbank @ power
            log_mel = np.log(mel_energy + 1e-10)
            mfcc = np.zeros(n_mfcc)
            for c in range(n_mfcc):
                for m_idx in range(n_mels):
                    mfcc[c] += log_mel[m_idx] * np.cos(np.pi * c * (m_idx + 0.5) / n_mels)
            mfcc_frames.append(mfcc)
        features.append(np.array(mfcc_frames))

    return np.array(features, dtype=np.float32)

def to_f32(model, path):
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    b = c.convert()
    open(path,'wb').write(b); print(f"  F32: {len(b)/1024:.1f}KB"); return b

def to_f16(model, path):
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations=[tf.lite.Optimize.DEFAULT]; c.target_spec.supported_types=[tf.float16]
    b = c.convert()
    open(path,'wb').write(b); print(f"  F16: {len(b)/1024:.1f}KB"); return b

def to_int8(model, rep, path):
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations=[tf.lite.Optimize.DEFAULT]
    def rep_gen():
        for i in range(min(500, len(rep))):
            yield [rep[i:i+1].astype(np.float32)]
    c.representative_dataset = rep_gen
    c.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type=tf.int8; c.inference_output_type=tf.int8
    b = c.convert()
    open(path,'wb').write(b); print(f"  INT8:{len(b)/1024:.1f}KB"); return b

def eval_tflite(buf, data, labels, name):
    interp = tf.lite.Interpreter(model_content=buf); interp.allocate_tensors()
    inp=interp.get_input_details()[0]; out=interp.get_output_details()[0]
    correct=0
    for i in range(len(data)):
        s=data[i:i+1].astype(np.float32)
        if inp['dtype']==np.int8:
            sc,zp=inp['quantization']; s=(s/sc+zp).astype(np.int8)
        interp.set_tensor(inp['index'],s); interp.invoke()
        o=interp.get_tensor(out['index'])
        if out['dtype']==np.int8:
            sc,zp=out['quantization']; o=(o.astype(np.float32)-zp)*sc
        if np.argmax(o[0])==labels[i]: correct+=1
    acc=correct/len(labels)*100
    print(f"  {name:8s}: {acc:.1f}%"); return acc

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model_dir",default="./models")
    p.add_argument("--data_dir",default="./processed")
    args=p.parse_args()
    with open(os.path.join(args.model_dir,"mfcc_config.json")) as f: cfg=json.load(f)
    print("Loading data...")
    test_a,test_l=load_dataset(args.data_dir,"test")
    train_a,_=load_dataset(args.data_dir,"train")
    te=compute_mfccs(test_a,cfg["n_mfcc"],cfg["hop_length"],cfg["n_fft"])
    tr=compute_mfccs(train_a,cfg["n_mfcc"],cfg["hop_length"],cfg["n_fft"])
    model=tf.keras.models.load_model(os.path.join(args.model_dir,"small_cnn.keras"))
    _,ka=model.evaluate(te,test_l,verbose=0); print(f"  Keras: {ka*100:.1f}%")
    print("Converting...")
    f32=to_f32(model,os.path.join(args.model_dir,"model_float32.tflite"))
    f16=to_f16(model,os.path.join(args.model_dir,"model_float16.tflite"))
    i8=to_int8(model,tr,os.path.join(args.model_dir,"model_int8.tflite"))
    print("Evaluating...")
    a32=eval_tflite(f32,te,test_l,"Float32")
    a16=eval_tflite(f16,te,test_l,"Float16")
    a8=eval_tflite(i8,te,test_l,"INT8")
    print(f"\n{'Variant':<10}{'Size(KB)':>10}{'Accuracy':>10}{'<100KB':>8}")
    print("-"*40)
    for n,b,a in [("Float32",f32,a32),("Float16",f16,a16),("INT8",i8,a8)]:
        sz=len(b)/1024; print(f"{n:<10}{sz:>9.1f} {a:>9.1f}% {'YES' if sz<100 else 'NO':>8}")
    print(f"\n  Size reduction: {(1-len(i8)/len(f32))*100:.0f}%  Accuracy drop: {a32-a8:.1f}pp")
    json.dump({"float32":{"size_kb":round(len(f32)/1024,1),"accuracy":round(a32,1)},
               "float16":{"size_kb":round(len(f16)/1024,1),"accuracy":round(a16,1)},
               "int8":{"size_kb":round(len(i8)/1024,1),"accuracy":round(a8,1)}},
              open(os.path.join(args.model_dir,"quantization_summary.json"),'w'), indent=2)
    print(f"\nNext: python 04_export_to_c_header.py --tflite_path {args.model_dir}/model_int8.tflite")

if __name__=="__main__": main()
