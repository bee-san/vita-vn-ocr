# The owner's games: test-only series and training-eligible titles

**Superseded in part (owner, 2026-10-14 13:25Z, `docs/reservations.md`).** The only test sets are now the CLANNAD
family, vita-test and psp-test, with their VNs and release partners (`eval/manifests/reserved.json` "test"). Every
other owner screenshot is training data, including the Memories Off series, Aiyoku no Eustia and 428 below and every
former `-TEST` folder whose VN is not in the test reservation. A `-TEST` suffix no longer decides a folder's role
(`realplus/owner_frames.folder_info`). The sections below keep the earlier rules as history.

The owner captures live screenshots of games on a PS Vita (owner, 2026-10-13 10:51Z, 13:10Z and 13:12Z).
This supersedes the 09:25Z and 09:37Z notes; STEINS;GATE and Ao no Kanata are not coming. planetarian did
come, in the owner's Google Drive drop (19:21Z), and is training data (see "The Drive drop of 2026-10-13").
The owner's folders are under `shared/private/owner-games/`. A `-TEST` suffix means test only.

**Test only.** The owner's frames of these titles go to new fresh native test sets and to on-console
overlay tests. Each series is reserved: nothing may be trained, tuned, calibrated, selected or rendered
from it, no font or render source may come from it, and its frames never go to Codex.

- **Memories Off** (PSP; v1152), with its whole series.
- **Aiyoku no Eustia** (Vita; v3770).
- **428** (PSP; v1299).

The owner asked at 13:10Z for the last two to be reserved before any harvest touched them; the reservation takes effect with this change, and no harvest has added rows of them since. The owner described them as new
and never seen anywhere. Existing pools do hold some of their lines, scaled from PC and other screenshots
(table below).

**Training-eligible** (`OWNER_TRAINING`). These may be labelled for training, re-checked against
`reserved_vns` at ingest. They are private frames, so they never go to Codex.

- Vita: CHAOS;HEAD NOAH (v22505), CHAOS;CHILD (v14018), Rewrite (v751), ATRI (v27448).
- PSP: Sharin no Kuni (v57), Toradora! Portable (v4137), Kikokugai (v183), Fate/EXTRA CCC (v12369),
  Togainu no Chi (v434), Otome wa Boku ni Koishiteru (v195), Mashiro-iro Symphony (v1552),
  Koi to Senkyo to Chocolate (v4028).

None of them is reserved under the rules below (checked on the 2026-10-13 dump). Owner frames of any
reserved title (the popular PSP titles the owner listed are) are never accepted as training.

Some frames may be Adrenaline captures: a PSP game upscaled to 960x544 on the Vita. The source
resolution is recorded per file.

## The Drive drop of 2026-10-13

The owner shared `vn-ocr-dataset-20261013.zip` on Google Drive at 19:21Z (2,819,705,393 bytes, sha256
`d2443837d339c2b69e34b29eea343007be76716eb6250ff664badd7b7e7b446e`; fetched with gdown 6.4.2, all 11,376
files verified by the drop's own `tools/verify_dataset.py`). It stays under `shared/private/` and never goes
to Codex or any third party. `synth/owner_drop.py` identified each game on the 2026-10-13 VNDB dump, checked
it (`reserved_vns`, the eval and development sets by VN and by release session, and a thumbnail leak check
against every protected screenshot) and hardlinked its frames into `owner-games/` with a per-frame
`manifest.jsonl` and the decision in `ingest.json`. The public table, without any frame name or hash, is
`data/manifests/owner_drop_20261013.json`.

All 3,421 full frames are native captures (960x544 Vita, 480x272 PSP; no Adrenaline upscale). None is a
pixel duplicate of another, of the realplus pool, of the fresh-native sets or of an eval manifest.

| folder | VN | frames | role | why |
|---|---|---:|---|---|
| psp-planetarian-v34 | v34 | 202 | training | owner 19:21Z and 20:47Z (below) |
| psp-sharin-no-kuni-v57 | v57 | 217 | training | not reserved; existing folder |
| vita-chaos-child-v14018 | v14018 | 159 | training | not reserved; existing folder |
| psp-amagami-v1281-TEST | v1281 | 113 | test | development VN |
| psp-narcissu-portable-v10-TEST | v10 | 200 | test | shares a release (ULJM-05674) with test-set VN v18738 |
| vita-clannad-v4-TEST | v4 | 2,206 | test | CLANNAD, protected; 2,204 of them are the CLANNAD long run |
| vita-danganronpa-v3-v18334-TEST | v18334 | 84 | test | shares a release with test-set VN v7014 |
| vita-iwaihime-matsuri-v17863-TEST | v17863 | 240 | test | development VN (5 teacher_val rows) |

That is 578 training frames and 2,843 test frames. No training frame reaches the leak threshold (0.95)
against a protected screenshot; the highest is planetarian's 0.941. Corpse Party BloodCovered (v2632) and
Grisaia no Kajitsu (v5154), both development VNs, have no full frames in the drop; Rewrite (v751) has two
quarantined intro images and no folder.

**planetarian (v34).** The owner made it training data in all its releases, even where it overlaps an eval
set (19:21Z, restated 20:47Z). `realplus/shots.OWNER_ALLOWED` keeps `reserved_vns` from giving it a derived
reason: it was reserved only as `shared-release`, through the PC Key 10th Memorial BOX (r3748) that also
holds CLANNAD. Direct reasons would still apply, and it has none. On the dump the only change is v34 itself
(18,846 to 18,845 reserved VNs). It has no rows in vita-test, psp-test, teacher_val, teacher_distill_val,
real_val or devnative v1-v3 (by VN and by release session), so no eval set needed a new version and every
existing score is unchanged. Its PSP port and CLANNAD's Vita port are both Key/Prototype releases, and VNDB
lists a character relation between planetarian and v5698 (CLANNAD: Lost "Winter", on the CLANNAD family
list; `reserved_vns` reserves a family member's family relations, not its character relations). A CLANNAD
score of any model trained on these frames must disclose that exposure.

## The Drive drop of 2026-10-14

The owner shared `VN-Screenshots-SFW-2026-10-14-5050.zip` on Google Drive (1,489,453,834 bytes, sha256
`6b4643776cbf1529f8d730db29753934bfa44f1eab6a7027caab67f05a3c7f5c`). It was fetched with gdown 6.4.2 (pinned, own venv)
into `shared/private/owner-drive-20261014/`, where a local manifest records every file's size and sha256. The drop's
`SHA256SUMS.txt` verifies all 5,228 files it lists, which is every file but itself. It stays under `shared/private/`
and never goes to Codex or any third party.

It holds 5,050 PNG frames of 87 games, at most 150 per game, in `games/<folder>/` with a per-frame `manifest.jsonl`:
- 4,963 dialogue frames (`visually_reviewed_new_dialogue`) and 87 title or settings screens;
- 3,071 PSP frames at 480x272 and 1,979 Vita frames at 960x544: every frame is at its platform's own screen size. A
  PSP game captured through Adrenaline would be 960x544, and none is. `owner_frames.upscale_2x` still reads 2 Vita
  frames as possible 2x upscales: VA-11 HALL-A's one frame (nearest-neighbour) and a Mikagami Sumika frame (smooth).
  Owner pool v2's `pool-prep` refuses both ([owner-pool](owner-pool.md#frames)).

`synth/owner_drop.py` checked every folder's VN on the 2026-10-13 VNDB dump (`GAMES_20261014`). The drop names a VN for
86 folders (`canonical_vndb_ids`), and the dump confirms each one:
- for 32 games, VNDB links the capture slot's console title id to a release of that VN, and the match is exact;
- for the other 54, the VN's title matches and it has a Japanese release on that platform.
The remaining folder, Tantei Jinguuji Saburo: Hai to Diamond, names no VN; a title search finds v19010 (PSP, ULJM-05529).
Roles follow the final policy (`decide`): a game is test-only when its VN is in the test reservation. Then:

- **Duplicates.** Every frame is compared by decoded-pixel sha256 with every frame already in owner-games and with the
  drop's earlier frames. 859 training frames are copies of the 2026-10-13 drop: Amagami 110, Narcissu 150, planetarian
  150, Sharin no Kuni 150, CHAOS;CHILD 150 and Iwaihime 149. They are listed but not linked again.
- **Leak check.** Each training frame's thumbnail (`eval/tools/leakcheck.thumb`) is compared with the three test sets
  only: vita-test's and psp-test's screenshots, the owner's CLANNAD frames and this drop's test-game frames. A frame at or
  above 0.95 is left out of its folder, without blocking its game. That drops 6 frames, all cross-VN: 3 Mikagami Sumika
  frames against CLANNAD frames (0.956), 1 MagusTale frame against a test-game frame (0.967), and 2 planetarian frames
  that were copies of the first drop anyway (0.951 and 0.952). The highest score against vita-test is 0.941, against
  psp-test 0.838.
- **Sort.** `sort` hardlinks each game's new frames into `owner-games/<platform>-<slug>-v<id>[-TEST]/` as
  `d20261014_<sequence>.png`, reusing a VN's existing folder. It writes `manifest_20261014.jsonl` (per frame: hashes,
  size, category, slot, whether skipped) and `ingest_20261014.json`. The first drop's files are untouched.
  `data/manifests/owner_drop_20261014.json` is the public table, with no frame name or hash.

| | games | frames | new (linked) |
|---|---:|---:|---:|
| test (the test reservation) | 4 | 408 | 408 |
| training | 83 | 4,642 | 3,779 (2,210 PSP, 1,569 Vita) |
| **all** | **87** | **5,050** | **4,187** |

The four test games are Time Travelers (v10856, PSP, 150 frames; a vita-test and psp-test game), Sweet Clown (v16460,
Vita, 1; vita-test), Utawarerumono (v3, PSP, 150; on the Trilogy Box with vita-test's v7721) and Collar x Malice (v18160,
Vita, 107; on the Twin Pack with vita-test's v21789). Their frames are test-only, like CLANNAD's: they were never read
or labelled, and serve only as the comparison side of the leak checks.

Training includes the Memories Off titles (v1170, v1171, v1325, v1646), which the final policy made training data.
It also includes five games that were in #81's development sample: No Fate! (v3278), Disgaea Infinite (v3333), Hanasaku
Manimani (v11677), Hyakka Yakou (v15773) and MISSING PARTS (v16386). Owner screenshots always train, so the sample was
re-carved without them (`docs/reservations.md`, "The re-carve after the owner's second drop").

| folder (`owner-games/`) | VN | title (VNDB) | platform | frames | of them menus | new | role | note |
|---|---|---|---|---:|---:|---:|---|---|
| psp-time-travelers-v10856-TEST | v10856 | タイムトラベラーズ | psp | 150 | 0 | 150 | test | psp-test / vita-test game |
| psp-utawarerumono-portable-v3-TEST | v3 | うたわれるもの | psp | 150 | 2 | 150 | test | shares a release with a test-set game |
| vita-collar-x-malice-v18160-TEST | v18160 | Collar x Malice | vita | 107 | 2 | 107 | test | shares a release with a test-set game |
| vita-sweet-clown-v16460-TEST | v16460 | SWEET CLOWN ~午前三時のオカシな道化師~ | vita | 1 | 1 | 1 | test | psp-test / vita-test game |
| psp-custom-drive-v10623 | v10623 | カスタムドライブ | psp | 150 | 2 | 150 | training |  |
| psp-da-capo-plus-situation-v264 | v264 | D.C.～ダ・カーポ～ | psp | 150 | 0 | 150 | training |  |
| psp-dunamis15-v7018 | v7018 | DUNAMIS15 | psp | 150 | 0 | 150 | training |  |
| psp-magustale-eternity-v689 | v689 | MagusTale ～世界樹と恋する魔法使い～ | psp | 150 | 2 | 149 | training | 1 leak-dropped |
| psp-memories-off-6-v1646 | v1646 | メモリーズオフ6 ～T-wave～ | psp | 150 | 0 | 150 | training |  |
| psp-memories-off-sorekara-v1171 | v1171 | Memories Off ～それから～ | psp | 150 | 0 | 150 | training |  |
| psp-narcissu-portable-v10-TEST | v10 | narcissu | psp | 150 | 0 | 0 | training | 150 already in owner-games |
| psp-no-fate-v3278 | v3278 | のーふぇいと！～only the power of will～ | psp | 150 | 0 | 150 | training |  |
| psp-pastel-chime-continue-v1080 | v1080 | ぱすてるチャイムContinue | psp | 150 | 2 | 150 | training |  |
| psp-planetarian-v34 | v34 | planetarian ～ちいさなほしのゆめ～ | psp | 150 | 0 | 0 | training | 150 already in owner-games; 2 leak-dropped; owner decision 2026-10-13 |
| psp-sharin-no-kuni-v57 | v57 | 車輪の国、向日葵の少女 | psp | 150 | 0 | 0 | training | 150 already in owner-games |
| psp-solfege-sweet-harmony-v2113 | v2113 | ソルフェージュ | psp | 150 | 0 | 150 | training |  |
| psp-suzukaze-no-melt-v3992 | v3992 | 涼風のメルト -Where wishes are drawn to each other- | psp | 150 | 2 | 150 | training |  |
| psp-tenshin-ranman-v1322 | v1322 | 天神乱漫 LUCKY or UNLUCKY!? | psp | 150 | 2 | 150 | training |  |
| psp-juusei-to-diamond-v14909 | v14909 | 銃声とダイヤモンド | psp | 149 | 1 | 149 | training |  |
| psp-amagami-v1281-TEST | v1281 | アマガミ | psp | 110 | 0 | 0 | training | 110 already in owner-games |
| psp-hakuisei-renai-re-therapy-v7301 | v7301 | 白衣性恋愛症候群 | psp | 100 | 0 | 100 | training |  |
| psp-missing-parts-complete-v16386 | v16386 | MISSING PARTS the TANTEI stories | psp | 49 | 0 | 49 | training |  |
| psp-trick-logic-v11713 | v11713 | Trick x Logic Season 1 | psp | 48 | 0 | 48 | training |  |
| psp-fate-extra-ccc-v12369 | v12369 | Fate/EXTRA CCC | psp | 44 | 1 | 44 | training |  |
| psp-danganronpa-v7014 | v7014 | ダンガンロンパ 希望の学園と絶望の高校生 | psp | 38 | 0 | 38 | training |  |
| psp-da-capo-ii-plus-situation-v266 | v266 | D.C.II ～ダ・カーポII～ | psp | 37 | 2 | 37 | training |  |
| psp-blazblue-cs-extend-v6572 | v6572 | BLAZBLUE CONTINUUM SHIFT | psp | 32 | 2 | 32 | training |  |
| psp-fate-extra-v7602 | v7602 | フェイト/エクストラ | psp | 24 | 2 | 24 | training |  |
| psp-jinguuji-saburo-hai-to-diamond-v19010 | v19010 | 探偵 神宮寺三郎 灰とダイヤモンド | psp | 11 | 2 | 11 | training |  |
| psp-princess-maker-5-portable-v728 | v728 | プリンセスメーカー5 | psp | 10 | 2 | 10 | training |  |
| psp-omoide-ni-kawaru-kimi-v1170 | v1170 | 想い出にかわる君 ～メモリーズオフ～ | psp | 6 | 0 | 6 | training |  |
| psp-memories-off-5-v1325 | v1325 | Memories Off #5 とぎれたフィルム | psp | 3 | 0 | 3 | training |  |
| psp-hayarigami-portable-v2063 | v2063 | 流行り神 警視庁怪異事件ファイル | psp | 2 | 2 | 2 | training |  |
| psp-princess-maker-4-portable-v727 | v727 | プリンセスメーカー4 | psp | 2 | 2 | 2 | training |  |
| psp-second-novel-v4091 | v4091 | セカンドノベル ～彼女の夏、15分の記憶～ | psp | 2 | 0 | 2 | training |  |
| chaos-head-noah-v22505 | v22505 | CHAOS;HEAD NOAH | psp | 1 | 0 | 1 | training |  |
| psp-disgaea-infinite-v3333 | v3333 | ディスガイア インフィニット | psp | 1 | 1 | 1 | training |  |
| psp-hayarigami-2-portable-v21259 | v21259 | 流行り神2 警視庁怪異事件ファイル | psp | 1 | 1 | 1 | training |  |
| psp-misshitsu-no-sacrifice-v4185 | v4185 | 密室のサクリファイス | psp | 1 | 1 | 1 | training |  |
| vita-chaos-child-v14018 | v14018 | CHAOS;CHILD | vita | 150 | 0 | 0 | training | 150 already in owner-games |
| vita-hakuisei-aijou-izonshou-v16610 | v16610 | 白衣性愛情依存症 | vita | 150 | 0 | 150 | training |  |
| vita-iwaihime-matsuri-v17863-TEST | v17863 | 祝姫 | vita | 150 | 1 | 1 | training | 149 already in owner-games |
| vita-steins-gate-v2002 | v2002 | STEINS;GATE | vita | 150 | 0 | 150 | training |  |
| vita-tokyo-ghoul-jail-v17936 | v17936 | 東京喰種 JAIL | vita | 120 | 2 | 120 | training |  |
| vita-amnesia-v-edition-v7803 | v7803 | AMNESIA | vita | 119 | 0 | 119 | training |  |
| vita-tsuki-ni-yorisou-otome-no-sahou-v10680 | v10680 | 月に寄りそう乙女の作法 | vita | 116 | 1 | 116 | training |  |
| vita-steins-gate-darling-v6618 | v6618 | STEINS;GATE 比翼恋理のだーりん | vita | 104 | 0 | 104 | training |  |
| vita-zero-escape-vlr-v7809 | v7809 | 極限脱出ADV 善人シボウデス | vita | 102 | 1 | 102 | training |  |
| vita-teikoku-kaigun-koi-bojou-v17285 | v17285 | 帝国海軍恋慕情 ～明治横須賀行進曲～ | vita | 88 | 2 | 88 | training |  |
| vita-robotics-notes-elite-v5883 | v5883 | ROBOTICS;NOTES | vita | 86 | 2 | 86 | training |  |
| vita-hyakka-yakou-v15773 | v15773 | 百華夜光 | vita | 79 | 2 | 79 | training |  |
| vita-ozmafia-vivace-v7833 | v7833 | OZMAFIA!! | vita | 76 | 1 | 76 | training |  |
| vita-side-kicks-v19743 | v19743 | Side Kicks! | vita | 63 | 2 | 63 | training |  |
| vita-princess-arthur-v11108 | v11108 | Princess Arthur | vita | 53 | 2 | 53 | training |  |
| vita-mikagami-sumika-v22174 | v22174 | 見鏡澄香の制服活動 | vita | 52 | 0 | 49 | training | 3 leak-dropped |
| vita-steins-gate-0-v17102 | v17102 | STEINS;GATE 0 | vita | 48 | 0 | 48 | training |  |
| vita-sakura-sou-no-pet-na-kanojo-v11173 | v11173 | さくら荘のペットな彼女 | vita | 38 | 1 | 38 | training |  |
| vita-xblaze-code-embryo-v13059 | v13059 | XBLAZE CODE:EMBRYO | vita | 33 | 2 | 33 | training |  |
| vita-steins-gate-phenogram-v11660 | v11660 | STEINS;GATE 線形拘束のフェノグラム | vita | 27 | 1 | 27 | training |  |
| vita-cross-ange-v19100 | v19100 | クロスアンジュ 天使と竜の輪舞tr. | vita | 13 | 0 | 13 | training |  |
| vita-chaos-head-love-chu-chu-v3091 | v3091 | カオスヘッド らぶChu☆Chu! | vita | 10 | 1 | 10 | training |  |
| vita-oregairu-v12630 | v12630 | やはりゲームでも俺の青春ラブコメはまちがっている。 | vita | 8 | 0 | 8 | training |  |
| vita-chouchou-jiken-rhapsodic-v19851 | v19851 | 蝶々事件ラブソディック | vita | 4 | 2 | 4 | training |  |
| vita-plastic-memories-v19441 | v19441 | プラスティック・メモリーズ | vita | 3 | 1 | 3 | training |  |
| vita-abunai-koi-no-sousashitsu-v16467 | v16467 | アブナイ恋の捜査室 ~Eternal Happiness~ | vita | 2 | 2 | 2 | training |  |
| vita-azayaka-na-irodori-v21899 | v21899 | あざやかな彩りの中で、君らしく | vita | 2 | 2 | 2 | training |  |
| vita-infinite-stratos-2-v13865 | v13865 | IS＜インフィニット・ストラトス＞2 イグニッション・ハーツ | vita | 2 | 2 | 2 | training |  |
| vita-moujuutsukai-to-oujisama-v4540 | v4540 | 猛獣使いと王子様 | vita | 2 | 2 | 2 | training |  |
| vita-kurogane-kaikitan-v17902 | v17902 | クロガネ回姫譚 -一期一会- | vita | 2 | 2 | 2 | training |  |
| vita-ray-gigant-v24957 | v24957 | レイギガント | vita | 2 | 2 | 2 | training |  |
| vita-5-nin-no-koi-prince-v16607 | v16607 | 5人の恋プリンス ~ヒミツの契約結婚~ | vita | 1 | 1 | 1 | training |  |
| vita-anoko-wa-ore-kara-hanarenai-v10271 | v10271 | 彼女はオレからはなれない | vita | 1 | 1 | 1 | training |  |
| vita-binary-star-v14787 | v14787 | Binary Star | vita | 1 | 1 | 1 | training |  |
| vita-cure-mate-club-v2923 | v2923 | Cure Mate Club | vita | 1 | 1 | 1 | training |  |
| vita-fate-extella-v19119 | v19119 | Fate/EXTELLA | vita | 1 | 1 | 1 | training |  |
| vita-hanasaku-manimani-v11677 | v11677 | 花咲くまにまに | vita | 1 | 1 | 1 | training |  |
| vita-hotchkiss-v7967 | v7967 | ホチキス | vita | 1 | 1 | 1 | training |  |
| vita-ikemen-bakumatsu-v18777 | v18777 | イケメン幕末◆運命の恋 | vita | 1 | 1 | 1 | training |  |
| vita-kissbell-v10641 | v10641 | キスベル | vita | 1 | 1 | 1 | training |  |
| vita-otoko-yuukaku-v15231 | v15231 | 逆転 吉原 | vita | 1 | 1 | 1 | training |  |
| vita-parfait-v683 | v683 | パルフェ ～ショコラ second brew～ | vita | 1 | 1 | 1 | training |  |
| vita-present-for-you-v5386 | v5386 | Present for you ～わたしをあ・げ・る～ | vita | 1 | 1 | 1 | training |  |
| vita-reine-des-fleurs-v14919 | v14919 | レンドフルール | vita | 1 | 1 | 1 | training |  |
| vita-shin-kamaitachi-no-yoru-v10561 | v10561 | 真かまいたちの夜 11人目の訪問者（サスペクト） | vita | 1 | 1 | 1 | training |  |
| vita-va-11-hall-a-v18872 | v18872 | VA-11 HALL-A | vita | 1 | 1 | 1 | training |  |
| vita-vamwolf-cross-v16468 | v16468 | Vamwolf Cross† | vita | 1 | 1 | 1 | training |  |
| vita-bitter-smile-v2954 | v2954 | Bitter Smile. | vita | 1 | 1 | 1 | training |  |

## What is reserved

`realplus/shots.owner_test_vns()` reserves 50 VNs in the 2026-10-13 VNDB dump:

- **The series** (40, reason `owner-test-series`): every VN reachable from a root (v1152, v3770, v1299)
  through VNDB's family relations (alternative versions, originals, fan discs, parent and side stories,
  sequels, prequels, same series), or titled like Memories Off (Memories Off, メモリーズオフ, 秋之回忆).
  Memories Off's series is 38 VNs. Several of them were on the training side until now: Your Memories Off
  ~Girl's Style~ (v1964), Memories Off Festa (v1966), Memories Off -Innocent Fille- (v21964) and its for
  Dearest (v23725), and most of the Chinese fan titles. Aiyoku no Eustia and 428 have no family relations.
- **Direct relations of any kind** of a series VN, in either direction (9, `owner-test-relation`).
  These are crossovers and same-setting titles:
  - for Memories Off: Never 7 (v248); Chaos;Head (v382), a character crossover with Memories Off 6;
    夢のつばさ (v5850); KID Mix Section (v26011); MemOff Mix (v27682); SINce Memories (v28683);
  - for 428: 街 (v1360), タイムトラベラーズ (v10856) and 3年B組金八先生 (v45398).
- **Release partners** (1, `owner-test-shared-release`): every VN on one VNDB release with a series VN.
  That is てんたま (v11517).

`owner_test_vns()` does not reserve a relation's own relations or release partners. CHAOS;HEAD NOAH is an
alternative version of Chaos;Head, not a relation of the series, and stays on the training side as the
owner asked.

`reserved_vns()` gives these reasons, so realplus, the native pool and every selection built on it leave
the 50 VNs out. Its general rule then reserves any VN on a release with one of them as `shared-release`,
as for the other protected titles. Against the rules before the owner's notes, 24 VNs are newly
reserved:

- 19 of Memories Off's 45-VN set (16 series VNs, 3 relations); the other 26 were already reserved for
  other reasons;
- Aiyoku no Eustia (v3770), 428 (v1299) and 街 (v1360). 428's other two relations were already reserved;
- 弟切草 (v661), which shares a PlayStation release with 街;
- Ever 17 - The Out of Infinity (v19373), which shares the Switch Ever17 / Never7 double pack with Never 7.
  Its rows in existing pools: teacher_train 11, teacher_distill_train 5, real_train 10, real_unlabelled 8,
  realplus v2 ink / GRAD 3 / 2.

Never 7's other compilation partners (Ever17, Remember11, 12RIVEN) were already reserved. synth3's manifest
holds the 50 VNs only, so v19373 and v661 stay usable for synth3's names and scenes. That is 22 name rows of
v661's characters and no local scene file.

Every pinned synth3 shard set and every pool in the table predates these rules. v3's names and scenes did
not exclude 428, Eustia or 街: 479 of v3's 1,000,000 train labels hold one of their markers. That is an
upper bound, since most of them are ordinary given names. Any score on the owner's fresh tests must
disclose this exposure.

`data/manifests/synth3_exclude_owner_games.json` (built by `synth/owner_games.py manifest`) is a synth3
exclusion manifest. synth3 renders and `synth3_stats.py` calibrations now use it by default, next to
the fresh-native and CLANNAD manifests. It holds:

- the 50 VNs, the series' relations with their kinds, and the release partners;
- 503 name hashes of the 593 VNDB characters of those VNs. A name row that hashes to one of them is
  never drawn;
- 730 text markers. A pool line or focus context that holds one is never drawn. The markers are every
  title, every two-part full name, and one-part names that hit at most 50 text-pool lines; 84 one-part
  names are common words and are listed but not used;
- the file and decoded-RGB sha256 of every local copy of a VNDB screenshot of those VNs (144 files),
  plus every file in an owner `*-TEST` folder when the manifest was built (none yet).

## What existing data already holds

The rules came after these pools were built. `synth/owner_games.py exposure` counts what they hold of
the 50 VNs, without text (`data/manifests/owner_games_exposure_20261013.json`). The exact row ids per
pool are kept locally in `$HERD/shared/artifacts/native-frames/owner_games_rows.json`.

| pool | rows | set rows | of which 428 / Eustia / 街 (v1299 / v3770 / v1360) |
|---|---:|---:|---|
| teacher_train | 12,115 | 123 | 14 / 2 / 41 |
| teacher_distill_train (unlabelled) | 9,044 | 122 | 2 / 2 / 21 |
| real_train | 9,811 | 111 | 14 / 4 / 23 |
| real_unlabelled | 11,174 | 127 | 3 / 2 / 36 |
| realplus v1 and v2, ink / GRAD | 71,153 / 53,566 (v2) | 8 / 6 (Innocent Fille for Dearest) | 0 |
| **teacher_val** (development) | 1,651 | **8** (v1331, Memories Off After Rain) | 0 |
| **devnative v3** (development): tv_native ink / GRAD, strict ink / GRAD | 513 / 423, 38 / 26 | **4 / 3, 4 / 3** (v1331) | 0 |
| real_val, teacher_distill_val | 1,495, 1,261 | 12, 10 (v1331) | 0 |

- teacher_val and devnative hold Memories Off After Rain, a development VN: any selection on them so
  far has touched a series title.
- synth3's scene candidates (realdata's shots) hold 65 PSP and 28 PC screenshots of the set (9 PSP shots of
  街, 4 PC shots each of 428 and Eustia). synth2's scene list holds 13 PSP ones.
- synth3's VNDB name pool has 437 train and 15 dev rows of the set's characters.
- 1,217 of synth3's ~3.8M text-pool lines hold a marker.

synth3 version 3's renders excluded v1152 and the series' test and development VNs from scenes, but not
the rest of the series.

## Reproduce

```sh
DB=$HERD/shared/artifacts/native-frames/vndb/dump/db          # vndb-db-2026-10-13 (realplus/shots.py dump)
.venv-ci/bin/python synth/owner_games.py manifest --dump $DB --out data/manifests/synth3_exclude_owner_games.json
.venv-ci/bin/python synth/owner_games.py exposure --manifest data/manifests/synth3_exclude_owner_games.json \
    --out data/manifests/owner_games_exposure_20261013.json
```

Rebuild the manifest when the owner's test frames land, so their file and pixel hashes are listed too.
