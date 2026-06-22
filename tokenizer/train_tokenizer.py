"""Train 256k BPE tokenizer on multilingual corpus."""
import sentencepiece as spm
import argparse, glob, os

def train(input_glob, output_dir, vocab_size=256000):
    files = sorted(glob.glob(input_glob))
    os.makedirs(output_dir, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=",".join(files[:5000]),  # sample
        model_prefix=f"{output_dir}/tokenizer",
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9999,
        num_threads=128,
        train_extremely_large_corpus=True,
        byte_fallback=True,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        normalization_rule_name="identity",
        user_defined_symbols=["<|endoftext|>", "<|im_start|>", "<|im_end|>",
                              "<|tool|>", "<|system|>", "<|user|>", "<|assistant|>"],
        max_sentence_length=1000000,
        shuffle_input_sentence=True,
        input_sentence_size=100000000,
    )
    print(f"Tokenizer saved to {output_dir}/tokenizer.model")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="glob of raw text files")
    p.add_argument("--output", default="tokenizer/out")
    p.add_argument("--vocab-size", type=int, default=256000)
    args = p.parse_args()
    train(args.input, args.output, args.vocab_size)
