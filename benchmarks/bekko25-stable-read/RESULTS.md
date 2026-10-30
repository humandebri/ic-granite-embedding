# Bekko a25m: 語彙表をstable memoryから直接読む試作

## レビュー対応後の再検証

既存 `load_from` の挙動を保ち、外部行モードの `embed` / `full` が panic ではなくエラーを返すよう修正した。検証ツールは Wasm の hash、heap の削減量、各 query の成功と bit 一致を確認し、失敗時は非ゼロ終了する。

修正版 raw Wasm `0ce0de4b515a543a16a47d6f9ac34cc057261ebf317066bbc7010e83c4a15ef9` を PocketIC 11.0.0 で旧版から upgrade し、同じ314ケースを再実行した。[再検証結果](verification-reviewed.json)では全ケースが成功・bit 一致し、heap は 568,000,512 → 174,653,440 B、stable memory は 418,447,360 B のままだった。観測した最大命令数は 4,803,805,418、旧版との差の最大は +709,827 命令。実 pack を使った別の API 検証では、通常読み込みと外部行モードの推論結果も bit 単位で一致した。

以下はレビュー対応前の試作 Wasm による元の測定記録である。以下の未測定事項は修正版にも当てはまる。

2026-10-30、PocketIC 11.0.0 / Application subnetで、現在の分割読み込み版と試作版を同じcanister・同じpackで比較した。隔離コピー上の試作であり、製品のcanisterコード・既存のローカルcanister・mainnetは変更していない。[全測定値](verification.json)・[試作コードと手順](../../experiments/bekko25-stable-read/README.md)

試作は行列演算用の重みをheapに保持し、256,000行×384個のF32語彙表だけをheapに確保しない。queryごとに、入力tokenにつき1回、対応する1,536 bytesをstable memoryから読み、通常のF32推論に渡す。最大256 tokenでは最大256回・393,216 bytes。重複tokenのcacheやreadのまとめは行っていない。固定packの検証済みhashと固定offsetを前提とした試作であり、一般化した安全な実装ではない。

## メモリ

| 指標 | 現行 | 試作 | 差 |
|---|---:|---:|---:|
| Wasm heap | 568,000,512 B | 174,653,440 B | **-393,347,072 B（-69.25%）** |
| stable memory | 418,447,360 B | 418,447,360 B | 0 B |
| heap + stable | 986,447,872 B | 593,100,800 B | **-39.88%** |

activate直後の `canister_status.memory_metrics` による永続メモリ。query中の一時メモリ最大値、Wasm本体や履歴を含む総memory sizeではない。13ノードsubnetの公開単価127,000 cycles/GiB/秒に単純換算すると、heap+stableの30日分は約0.3024T→0.1818T cycles。その他の領域、操作費、snapshot、メモリ割当は含まない。[ICP料金表](https://docs.internetcomputer.org/references/cycle-costs/)

## Query命令数と出力

旧版をインストールしpackを一度だけ投入して測定後、試作へupgrade。verified重みが残ってinactiveとなること、非controllerのactivate拒否、controllerの再activate、stable容量の一致を確認した。2〜256 tokens各長255ケース、6言語×3長18ケース、日本語40ケース、256-token繰り返し1ケースの計314ケースがすべて成功し、出力embeddingは**全ケースbit単位で一致**した。

| tokens | 現行の命令数 | 試作の命令数 | 差 |
|---:|---:|---:|---:|
| 2 | 46,958,890 | 46,955,669 | -3,221（-0.0069%） |
| 64 | 1,034,825,580 | 1,034,910,804 | +85,224（+0.0082%） |
| 128 | 2,199,980,668 | 2,200,138,362 | +157,694（+0.0072%） |
| 192 | 3,484,375,050 | 3,484,645,807 | +270,757（+0.0078%） |
| 255 | 4,803,464,245 | 4,803,805,418 | +341,173（+0.0071%） |
| 256 | 4,792,954,656 | 4,793,296,646 | +341,990（+0.0071%） |

表は固定seedの各長1入力。全314ケースでは291増・23減、最大増加は71-tokenケースの709,827命令（+0.0381%）。観測最大は4,803,805,418 / 5,000,000,000命令。元の別stress runの最大値4,843,751,277は今回の入力に含まれず、この試作の全入力での上限は未証明。

activateの壁時計は現行3.07秒、試作0.18秒だったが、それぞれ1回のローカル観測であり、起動速度やmainnet遅延の保証ではない。query命令数は `performance_counter` による値であり、ネットワーク往復や実際の主記憶・ページキャッシュ差を含むmainnetの応答時間ではない。

- 現行 raw Wasm SHA-256: `c198d9ade07132daf4c9a27151744b9381bb7ebfbbf149c6829bf7f70046cede`
- 試作 raw Wasm SHA-256: `893e615848710bbe4cc2f1e3c99b51d943a6ac46bcfa2b0854ecc2374092f2bc`
- pack SHA-256: `8fc88d56d68f550d1a920bab7d1ec42af9449240048393bd6daabbfcfa861724`

recipe処理後Wasm、mainnet、より多い入力、queryの一時メモリ、同時利用時の応答時間は未測定。実装に進む場合はpack metadataからoffsetを検証し、既存のstorage schema・upgrade・入力拒否・Candid互換性を保持したうえで、実Wasmの再現性、全入力長とstress、実文書retrieval、queryのメモリ・遅延を検証する。
