# 命令数優先版を既定に採用

> 当時の専用runner・実験featureは整理済みです。再実行には[現行の開発手順](../../docs/DEVELOPMENT.md)を使ってください。

> **履歴: このページは当時の測定記録です。** 本文の上限・query数・「既定」は当時の構成を指します。現在の仕様と結果は[最新レポート](../frontier/RESULTS.md)、経緯は[計測一覧](../README.md)を参照してください。

2026-10-27。`instruction-gemm`をcanisterのdefault featuresへ追加した。
256 tokensは4,230,646,590命令、288 tokensまで1 query、512 tokensは2 query。
ローカル中央値0.476秒→0.566秒という実行時間の増加を受け入れ、命令数削減を優先する。

新しい既定ビルドのSHA256は`faa1c76da109a72f59cddef91e2e889c6c471d7b2cf65188a443ec44c3d27121`。
前回、全511長・追加57ケース・公開API 16ケースで検証した命令数優先版とバイト単位で同一だった。
このため推論の全長検証結果を引き継ぎ、今回はビルド一致とunit test 10件を確認した。
標準の`target/wasm32-unknown-unknown/release/granite_canister.wasm`にも同じバイナリを配置した。

[採用時の設定・source SHA・検証の由来](adoption.json)、[既存の全長検証](../tile-optimization/instruction-gemm/environment.json)、[実測比較](../tile-optimization/RESULTS.md)。

次の候補は、命令数の削減を保ちながら、生成kernelのコード量と同時に保持するSIMD値を減らすこと。
CPUのどこで時間が増えたかを先に計測し、実行時間の悪化を抑える。これは今後の方針で、未検証の改善である。
