| Rank | System | Track | Acc ↑ | Macro-F1 ↑ | ECE ↓ | Latency (ms, median) ↓ | CER ↓ |
|---|---|---|---|---|---|---|---|
| 1 | Whisper-large-v3-turbo → KLUE-RoBERTa | Cascade | 0.8532 | 0.8066 | 0.0244 | 245.0 | 0.0528 |
| 2 | Whisper-small → KLUE-RoBERTa | Cascade | 0.8273 | 0.7814 | 0.0361 | 208.6 | 0.0995 |
| 3 | Whisper-small encoder + attentive pooling | Direct | 0.6633 | 0.5811 | 0.0728 | 39.9 | — |
| ref | Gold transcript → KLUE-RoBERTa | Oracle | 0.8791 | 0.8234 | 0.0377 | 9.1 | — |
