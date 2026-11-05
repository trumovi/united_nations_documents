import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from text import Text
import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = r"C:\Users\user\folder"

for subdir, _, files in os.walk("un_texts"):
    for file in files:
        if file == "doc_3_ru.txt":
            filepath = os.path.join(subdir, file)
            print(file)
            text = Text(filepath)

            selected_words, selected_count, target_count, indicators, bridge = \
                text.empirical_bridge()

            print("Всего слов:", len(text.words))
            print("Целевое количество вхождений:", target_count)
            print("Количество выбранных слов:", selected_count)

            print("\nВыбранные слова:")
            for word in selected_words:
                print(word)

            print("\nИндикаторы:")
            print(indicators)

            print("\nЭмпирический мост:")
            print(bridge)

            import matplotlib.pyplot as plt


            def plot_empirical_bridge(text, name, output_dir=OUTPUT_DIR):
                (
                    selected_words,
                    selected_count,
                    target_count,
                    indicators,
                    bridge
                ) = text.empirical_bridge()

                N = len(text.words)
                sqrt_n = np.sqrt(N)
                bridge = np.asarray(bridge, dtype=float) / sqrt_n

                plt.figure(figsize=(12, 5))

                plt.plot(
                    np.arange(1, N + 1),
                    bridge
                )

                plt.axhline(
                    0,
                    linestyle='--',
                    linewidth=1
                )

                plt.xlabel("Позиция слова в тексте")
                plt.ylabel("Эмпирический мост")
                plt.title(
                    f"Эмпирический мост по индикаторам "
                    f"{selected_count} наиболее частых слов "
                )

                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                #plt.show()
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, name + "_bridge.png")
                plt.savefig(out_path)
                #plt.savefig(name + ".png")


            name = file.rstrip(".txt")
            plot_empirical_bridge(text, name)
