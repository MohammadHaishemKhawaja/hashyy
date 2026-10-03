import urllib.request,zipfile,os
D=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llama"); os.makedirs(D,exist_ok=True)
B="https://github.com/ggml-org/llama.cpp/releases/download/b11320/"
for n in ["llama-b11320-bin-win-cuda-13.4-x64.zip","cudart-llama-bin-win-cuda-13.4-x64.zip"]:
    p=os.path.join(D,n)
    if not os.path.exists(p):
        print("downloading",n,flush=True); urllib.request.urlretrieve(B+n,p)
    with zipfile.ZipFile(p) as z: z.extractall(D)
    print("extracted",n,flush=True)
    os.remove(p)
print("EXES:",sorted(f for f in os.listdir(D) if f.endswith('.exe'))[:30])
