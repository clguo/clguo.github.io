"""Verify publication links and saved evidence without another GPU evaluation."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote,urlparse
import subprocess
import zipfile
import torch
from PIL import Image

class Links(HTMLParser):
    def __init__(self):super().__init__();self.refs=[];self.ids=set();self.videos=0
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        for key in ["href","src","poster"]:
            if key in a:self.refs.append(a[key])
        if "id" in a:self.ids.add(a["id"])
        self.videos+=tag=="video"

def main():
    torch.set_num_threads(2)
    report_path=Path("results/unified/evaluation/report.json")
    report=json.loads(report_path.read_text())
    assert all(report["audit_checks"].values())
    for page in ["index.html","comparison.html","repository_audit.html"]:
        text=Path(page).read_text();parser=Links();parser.feed(text)
        assert "PLACEHOLDER" not in text and "@@RESULTS@@" not in text
        for ref in parser.refs:
            parsed=urlparse(ref)
            if parsed.scheme or parsed.netloc:continue
            if not parsed.path:
                if parsed.fragment:assert unquote(parsed.fragment) in parser.ids,(page,ref)
            else:assert Path(unquote(parsed.path)).exists(),(page,ref)
        if page!="repository_audit.html":assert parser.videos==1
        print(page,"links, anchors, equations and video passed")
    visual=torch.cat([torch.arange(10)+100*k for k in range(10)])
    bank=torch.load("results/unified/evaluation/input_bank.pt",weights_only=True)
    assert torch.equal(bank["visual_indices"],visual)
    assert torch.equal(bank["labels"].bincount(),torch.full((10,),100))
    for key in ["ddpm","ddim","fm","rf2","continued"]:
        endpoints=torch.load(f"results/unified/evaluation/{key}_endpoints.pt",weights_only=True)
        path=torch.load(f"results/unified/evaluation/{key}_trajectory.pt",weights_only=True)
        assert path.shape==(65,100,1,28,28)
        assert torch.equal(path[0],bank["noise"][visual])
        assert torch.allclose(path[-1],endpoints["64"][visual],atol=1e-6,rtol=1e-6)
        for nfe,row in report["configurations"][key].items():
            if nfe=="dense_path":continue
            assert endpoints[nfe].shape==(1000,1,28,28)
            assert torch.isfinite(endpoints[nfe]).all()
            assert row["counted_network_calls"]==10*int(nfe)
        print(key,"unfiltered endpoints, counted calls and visual replay passed")
    for stage,row in report["checkpoints"].items():
        p=Path("results/unified")/stage/"checkpoint.pt"
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row["sha256"]
        meta=row["metadata"]
        assert meta["steps"]==meta["planned_steps"]
    with Image.open("assets/unified_generation.gif") as gif:
        assert gif.n_frames==33
        gif.seek(32)
        assert gif.info["duration"]>=1500
        print("GIF",gif.n_frames,"frames",gif.size)
    video=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_streams","-of","json","assets/unified_generation.mp4"]))
    assert video["streams"][0]["codec_name"]=="h264"
    print("H.264 video, checkpoint hashes, completed budgets: passed")
    with zipfile.ZipFile("mnist-generative-models-unified.zip") as archive:
        assert archive.testzip() is None
        members=set(archive.namelist())
        for page in ["index.html","comparison.html","repository_audit.html"]:
            parser=Links();parser.feed(archive.read(page).decode())
            for ref in parser.refs:
                parsed=urlparse(ref)
                if not parsed.scheme and not parsed.netloc and parsed.path:
                    assert unquote(parsed.path) in members,(page,ref)
    print("Publication ZIP integrity and all bundled local references: passed")

if __name__=="__main__":main()
