"""Independent intelligibility check; ASR is not a substitute for a listening review."""
import argparse
import json
import re
from pathlib import Path
from faster_whisper import WhisperModel


def words(text):
    return re.findall(r"[a-z]+", text.lower().replace("can't", "cannot").replace("it's", "it is"))


def distance(a, b):
    row = list(range(len(b)+1))
    for i,x in enumerate(a):
        next_row = [i+1]
        for j,y in enumerate(b):
            next_row.append(min(row[j+1]+1,next_row[j]+1,row[j]+(x!=y)))
        row = next_row
    return row[-1]


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('artifacts',type=Path);p.add_argument('--model',required=True)
    p.add_argument('--profiles', nargs='+', default=['fp32', 'int4'])
    args=p.parse_args()
    model=WhisperModel(args.model,device='cpu',compute_type='int8',cpu_threads=2)
    results=json.loads((args.artifacts/'asr-validation.json').read_text()) if (args.artifacts/'asr-validation.json').exists() else []
    for profile in args.profiles:
        records=json.loads((args.artifacts/f'validation-{profile}.json').read_text())
        for record in records:
            if any(r['profile']==profile and r['name']==record['name'] for r in results): continue
            path=args.artifacts/'samples'/profile/(record['name']+'.wav')
            segments,_=model.transcribe(str(path),language='en',beam_size=5,vad_filter=False)
            text=' '.join(s.text.strip() for s in segments)
            expected=words(record['text']); actual=words(text)
            result=dict(profile=profile,name=record['name'],transcript=text,wer=distance(expected,actual)/len(expected))
            results.append(result);print(json.dumps(result),flush=True)
    (args.artifacts/'asr-validation.json').write_text(json.dumps(results,indent=2))
