"""Build the focused homepage/source bundle; no external code or model weights."""
import json
import zipfile
from pathlib import Path

def main():
    root=Path.cwd()
    files=[Path(name) for name in ["README.md","requirements.txt","blog.md","comparison.md","repository_audit.md","index.html","comparison.html","repository_audit.html","digit_model.py","quality.py","train.py","build_article.py","optimize_assets.py","unified_tests.log"]]
    files+=sorted(Path("unified").glob("*.py"))+[Path("unified/protocol.json"),Path("unified/tutorial.md")]
    files+=sorted(Path("assets").glob("unified_*"))
    files+=[Path("results/unified/evaluation/report.json")]
    for stage in ["diffusion","flow","pairs","reflow","continued"]:
        files+=[Path("results/unified")/stage/"metadata.json"]
        if stage!="pairs":files+=[Path("results/unified")/stage/"loss.npy"]
    files+=sorted(Path("references").glob("*/snapshot.json"))+[Path("references/assignment_counterexample.json")]
    target=Path("mnist-generative-models-unified.zip")
    with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(set(files)):
            assert path.is_file(),str(path)
            archive.write(path,path.relative_to(root) if path.is_absolute() else path)
    print(target,round(target.stat().st_size/2**20,2),"MiB",len(set(files)),"files")

if __name__=="__main__":main()
