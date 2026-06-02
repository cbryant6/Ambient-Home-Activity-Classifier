"""
Stage 4: Export TFLite to C Header
Usage: python 04_export_to_c_header.py --tflite_path ./models/model_int8.tflite
"""
import os, argparse, json

CLASS_NAMES = ["quiet","human_activity","appliance_mechanical","transient_impact","environmental_ambient"]
DISPLAY_LABELS = {"quiet":"Quiet","human_activity":"Human","appliance_mechanical":"Appliance",
                  "transient_impact":"Impact","environmental_ambient":"Ambient"}
DISPLAY_COLORS = [("0xFFFF","white - quiet"),("0x07FF","cyan - human"),("0xFFE0","yellow - appliance"),
                  ("0xF800","red - impact"),("0x07E0","green - ambient")]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tflite_path",default="./models/model_int8.tflite")
    p.add_argument("--config_path",default="./models/mfcc_config.json")
    p.add_argument("--output_dir",default="./arduino_deploy")
    args=p.parse_args()
    os.makedirs(args.output_dir,exist_ok=True)

    # Model header
    b=open(args.tflite_path,'rb').read()
    with open(os.path.join(args.output_dir,"model_data.h"),'w') as f:
        f.write(f"// {os.path.basename(args.tflite_path)} — {len(b)/1024:.1f} KB\n")
        f.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n#include <cstdint>\n\n")
        f.write(f"const unsigned int model_data_len = {len(b)};\n")
        f.write("alignas(8) const unsigned char model_data[] = {\n")
        for i in range(0,len(b),12):
            chunk=b[i:i+12]
            f.write("  "+", ".join(f"0x{x:02x}" for x in chunk)+(",\n" if i+12<len(b) else "\n"))
        f.write("};\n#endif\n")
    print(f"  model_data.h written ({len(b)/1024:.1f} KB)")

    # Class config header
    if os.path.exists(args.config_path):
        cfg=json.load(open(args.config_path))
        with open(os.path.join(args.output_dir,"class_config.h"),'w') as f:
            f.write("#ifndef CLASS_CONFIG_H\n#define CLASS_CONFIG_H\n\n")
            f.write(f"#define NUM_CLASSES {cfg['n_classes']}\n")
            f.write(f"#define SAMPLE_RATE {cfg['sample_rate']}\n")
            f.write(f"#define N_MFCC {cfg['n_mfcc']}\n")
            f.write(f"#define HOP_LENGTH {cfg['hop_length']}\n")
            f.write(f"#define N_FFT {cfg['n_fft']}\n")
            f.write(f"#define CLIP_DURATION_MS {int(cfg['clip_duration']*1000)}\n\n")
            f.write("const char* CLASS_NAMES[NUM_CLASSES] = {\n")
            for i,n in enumerate(cfg['class_names']):
                f.write(f'  "{DISPLAY_LABELS.get(n,n)}"{"," if i<len(cfg["class_names"])-1 else ""}\n')
            f.write("};\n\nconst uint16_t CLASS_COLORS[NUM_CLASSES] = {\n")
            for i,(c,cm) in enumerate(DISPLAY_COLORS):
                f.write(f"  {c}{',' if i<len(DISPLAY_COLORS)-1 else ''}  // {cm}\n")
            f.write("};\n\n#endif\n")
        print("  class_config.h written")
    print(f"\nCopy model_data.h and class_config.h into your Arduino sketch folder")

if __name__=="__main__": main()
