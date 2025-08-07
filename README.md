# LoRA & DoRA in TinyGrad

This project demonstrates how to implement and apply **LoRA (Low-Rank Adaptation)** and **DoRA (Direct Output Rank Adaptation)** techniques in [TinyGrad](https://github.com/geohot/tinygrad). These methods allow efficient fine-tuning of deep learning models by injecting low-rank adapters into linear layers.

---

## 📊 View Computation Graph

To view the computation graph of the model:

```bash
GRAPH=1 ./test.py
```

---

## 🧠 What are LoRA & DoRA?

- **LoRA** allows you to fine-tune only a small number of trainable parameters by inserting low-rank matrices into pre-trained models, preserving the original weights.
- **DoRA** is a more recent technique that directly adapts the output ranks for efficient fine-tuning with minimal overhead.

Watch [Miss Coffee Bean's video](https://www.youtube.com/@misscoffeebean) for a friendly explanation and motivation behind these approaches.

---

## 📂 Project Structure

```
.
├── dora_tinygrad/           # DoRA implementation
│   └── modules/             # Base and linear module classes
├── lora_tinygrad/           # LoRA implementation
│   └── modules/             # Base and linear module classes
├── examples/                # Example scripts and utils
│   ├── example_lora.py      # End-to-end LoRA training + finetuning
│   ├── example_dora.py      # End-to-end DoRA training + finetuning
│   ├── mnist_example.ipynb  # Notebook to play with MNIST + LoRA
│   ├── test_lora.py         # Graph/debugging script for LoRA
│   └── utils.py             # Training, evaluation, misc helpers
├── test.py                  # Entry point for testing LoRA/DoRA
└── README.md                # You're here!
```

---

## 🚀 How to Run

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/tinygrad-lora-dora
cd tinygrad-lora-dora
```

### 2. Set up Virtual Environment (Recommended)

```bash
python -m venv .env
source .env/bin/activate
```

> Jupyter issues? Run: `ipython kernel install --name "local-venv-kernel" --user`

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Try the Examples

### Run LoRA Example:

```bash
python examples/example_lora.py
```

### Run DoRA Example:

```bash
python examples/example_dora.py
```

---

## 📘 Notes

- This project **does not use external libraries** like `peft`, `transformers`, or `accelerate`. It is meant to be educational and minimal.
- TinyGrad is a great environment to understand low-level ML concepts. We leverage this simplicity to explain and explore LoRA and DoRA directly in the core logic.

---

## 📄 References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [DoRA Paper](https://arxiv.org/abs/2402.09353)
- [TinyGrad by George Hotz (geohot)](https://github.com/geohot/tinygrad)

---

## 🧊 License

MIT License. See `LICENSE` file for details.

---

Happy fine-tuning with less memory! 🎉
