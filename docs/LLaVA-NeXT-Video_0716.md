## LLaVA-NeXT-Video is upgraded 🚀

In our [LLaVA-Video blog](https://llava-vl.github.io/blog/2024-04-30-llava-next-video/) released this April, we shared two key observations: 
- 🎬 AnyRes provides a shared and flexible representation between images and videos, and thus accommodates capability transfer between the two most common vision signals. Therefore, stronger image LMMs can naturally lead to stronger zero-shot video LMMs. 
- 🗂️ There is a lack of high-quality language-video data, including video instruction-following data, and thus naive tuning on existing public data at that time results in performance degradation. Therefore, there is an urgent need to build high-quality video captions and QA datasets to train LMMs for improved video performance.

Based on the insights, the new LLaVA-NeXT-Video in this release improves from two aspects:

- 🎬 A stronger image LMMs ([LLaVA-NeXT-32B-Qwen](https://huggingface.co/lmms-lab/llava-next-qwen-32b)), which is built by initializing from Qwen-1.5 32B LLM. We further initialize our video training from this image checkpoint.
- 🗂️ A new high-quality video dataset with 830k samples. It is combined with LLaVA-1.6 image training data, and applying the same image-video mixed training procedure leads to the new video model.
The new model achieves the best open-source performance in several video benchmarks including [Video-MME](https://video-mme.github.io/home_page.html#leaderboard).

### Resources
- **Model Card**: [LLaVA-NeXT-Video-32B-Qwen on Hugging Face](https://huggingface.co/lmms-lab/LLaVA-NeXT-Video-32B-Qwen)
- **Inference Script**:
  ```bash
  bash scripts/video/demo/video_demo.sh lmms-lab/LLaVA-NeXT-Video-32B-Qwen qwen_1_5 32 2 average grid True playground/demo/xU25MMA2N4aVtYay.mp4
  ```

### Evaluation Results
| Model                       | NextQA-MC | video-mme(overall) |        | Egochema | Perception Test  (val) |
|-----------------------------|-----------|--------------------|--------|----------|------------------------|
|                             |           | w/o subs           | w subs |          |                        |
| **Proprietary**                 |           |                    |        |          |                        |
| GPT-4o                      | -         | 71.9               | 77.2   | 72.2     | -                      |
| Gemini 1.5 Pro              | -         | 75.0               | 81.3   | 72.2     | -                      |
| **Open-Source**                 |           |                    |        |          |                        |
| VideoLLaMA 2 (8x7B)         | 76.3*     | 47.9               | 50.3   | 53.3     | 51.2*                  |
| VILA-1.5-34B                | 67.89*    | 60.1               | 61.1   | 58.04*   | 54                     |
| LLaVA-NeXT-Video (Qwen-32B) | 77.31     | 60.2               | 63.0   | 60.85    | 59.38                  |

_*Results are reproduced by [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). Please refer to the lmms-eval to reproduce the results._

### Citations
```bibtex
@misc{zhang2024llavanextvideo,
  title={LLaVA-NeXT: A Strong Zero-shot Video Understanding Model},
  url={https://llava-vl.github.io/blog/2024-04-30-llava-next-video/},
  author={Zhang, Yuanhan and Li, Bo and Liu, haotian and Lee, Yong jae and Gui, Liangke and Fu, Di and Feng, Jiashi and Liu, Ziwei and Li, Chunyuan},
  month={April},
  year={2024}
}
