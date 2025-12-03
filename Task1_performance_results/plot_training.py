import re
import os
import matplotlib.pyplot as plt


LOG_PATH = 'Task1_performance_results/training_log.txt'
OUT_DIR = 'Task1_performance_results'


def parse_training_log(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # split into stages
    stage_blocks = re.split(r'===== TRAINING (\d+)-stage =====', text)
    # re.split will produce list like [pre, stageNum1, block1, stageNum2, block2, ...]
    stages = {}
    for i in range(1, len(stage_blocks), 2):
        stage_num = int(stage_blocks[i])
        block = stage_blocks[i+1]
        # find table rows like: |   1|        1|        1|
        rows = re.findall(r'\|\s*(\d+)\|\s*([0-9.]+)\|\s*([0-9.]+)\|', block)
        entries = [(int(n), float(hr), float(fa)) for (n, hr, fa) in rows]
        stages[stage_num] = entries
    return stages


def plot_stages(stages, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # prepare plot: TPR and FPR on same axes but different styles
    plt.figure(figsize=(10,6))

    colors = ['C0','C1','C2','C3','C4']
    for idx, (stage, entries) in enumerate(sorted(stages.items())):
        Ns = [e[0] for e in entries]
        HRs = [e[1] for e in entries]
        FAs = [e[2] for e in entries]
        color = colors[idx % len(colors)]
        plt.plot(Ns, HRs, marker='o', color=color, linestyle='-', label=f'Stage {stage} TPR')
        plt.plot(Ns, FAs, marker='x', color=color, linestyle='--', label=f'Stage {stage} FPR')

    plt.title('Training progress: TPR (HR) and FPR (FA) per stage')
    plt.xlabel('Weak classifier index (N)')
    plt.ylabel('Rate')
    plt.ylim(-0.02, 1.05)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    out_png = os.path.join(out_dir, 'training_stages.png')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    return out_png


def write_analysis(stages, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_txt = os.path.join(out_dir, 'training_analysis.txt')
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write('Training stages analysis\n')
        f.write('========================\n\n')
        for stage, entries in sorted(stages.items()):
            Ns = [e[0] for e in entries]
            HRs = [e[1] for e in entries]
            FAs = [e[2] for e in entries]
            f.write(f'Stage {stage}: {len(entries)} weak classifiers recorded\n')
            f.write(f'  - Final TPR (HR): {HRs[-1]:.3f}\n')
            f.write(f'  - Final FPR (FA): {FAs[-1]:.3f}\n')
            f.write(f'  - TPR range: {min(HRs):.3f} to {max(HRs):.3f}\n')
            f.write(f'  - FPR range: {min(FAs):.3f} to {max(FAs):.3f}\n')
            f.write('\n')

    return out_txt


def main():
    if not os.path.exists(LOG_PATH):
        print(f'Log file not found: {LOG_PATH}')
        return
    stages = parse_training_log(LOG_PATH)
    if not stages:
        print('No stage data parsed from log')
        return
    png = plot_stages(stages, OUT_DIR)
    txt = write_analysis(stages, OUT_DIR)
    print('Wrote:', png)
    print('Wrote:', txt)


if __name__ == '__main__':
    main()
