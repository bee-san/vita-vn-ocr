#!/usr/bin/env python3
"""The owner's Google Drive drops (DROPS: 2026-10-13 and 2026-10-14): identify, check and sort.

  owner_drop.py check  --drop DIR [--dump DB] [--leak] --out PLAN.json
  owner_drop.py sort   --plan PLAN.json [--dest DIR]
  owner_drop.py public --plan PLAN.json --out JSON

The drops are private and copyrighted. They stay under shared/private/, never in git, and never go to Codex or
any other third party. Two layouts:
  samples  2026-10-13 (vn-ocr-dataset-20261013.zip; owner 19:21Z and 19:27Z): samples.jsonl lists every full
           frame (title, platform, image, dimensions, sha256); vita/CLANNAD/training-data/ is the CLANNAD long run
           (eval/clannad_longrun.py's session, with the published pins of eval/vocr3_clannad.py).
  games    2026-10-14 (VN-Screenshots-SFW-2026-10-14-5050.zip): games/<folder>/manifest.jsonl per game (image,
           sha256, dimensions, sequence, source run and slot, "category" title_screen / settings_screen for the
           menu frames; the dialogue frames are visually_reviewed_new_dialogue) and dataset-index.json (folder,
           source slot: the console title id, the drop's canonical VNDB ids). GAMES identifies every folder again on
           the 2026-10-13 dump: by the title id where a VNDB release links it (extlinks, catalog), else by title,
           platform and the VN's Japanese release on that platform; the drop's ids are only a cross-check.
For the games layout, check also marks each frame that is a pixel copy of a frame already in owner-games (or of
an earlier frame of the drop) as duplicate_of, and with --leak tests each frame against the three test sets only
(vita-test's and psp-test's screenshots and the owner's CLANNAD frames, test_set_shots): a frame at or above
MIN_CORR is left out of its training folder (leak_dropped), instead of blocking its game. sort never links a
duplicate or a leak-dropped frame, and names the frames d<date>_<file> next to the first drop's.

check   per full frame: the decoded size (the source resolution; every frame of this drop is a native
        capture, 960x544 Vita or 480x272 PSP, never an Adrenaline upscale) and three hashes (the file's, PIL's
        decoded RGB as the fresh-native harness hashes it, devnative's). Per game: its VNDB id (GAMES:
        identified on VNDB's 2026-10-13 dump by title, platform and the Japanese console release),
        realplus/shots.reserved_vns's reason on the dump, the VN's rows in eval's and the development sets
        (vita-test, psp-test, test_games.json "all", is_test_vn, teacher_val, devnative: information, apart from
        the two test sets) and, with --leak, each frame's best thumbnail correlation (eval/tools/leakcheck.thumb)
        with every screenshot of eval's candidates, devnative's cohort and teacher_val. decide() gives each game a
        role (owner 2026-10-14 13:25Z, docs/reservations.md: all owner screenshots train unless the game is a test
        game):
          test      CLANNAD and its family; else a VN of the test reservation (a vita-test or psp-test game or a
                    release partner of one) or a VN with rows in vita-test or psp-test;
          training  everything else, including a development-sample VN (then re-run realplus/shots.py reserve,
                    which leaves owner games out of the sample) and an owner decision (OWNER_DECISIONS);
          blocked   an owner decision whose VN is in vita-test or psp-test, or a training game with a frame at or
                    above MIN_CORR against a protected screenshot.
sort    hardlinks each game's frames into owner-games/<platform>-<slug>-v<id>[-TEST]/ (an existing folder of
        the same VN and role is reused) with manifest.jsonl (per frame: file, drop path, hashes, size, source
        resolution, sequence) and ingest.json (the decision); the 2026-10-14 drop writes manifest_20261014.jsonl
        and ingest_20261014.json, so a folder the first drop made keeps its files. It never overwrites or
        removes a file.
public  the per-game table without any frame's path or hash (data/manifests/owner_drop_<date>.json).
"""
import argparse
import collections
import datetime
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (HERE, ROOT / "realplus"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import shots as SH  # noqa: E402

from common import is_test_vn  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
OWNER_GAMES = HERD / "shared" / "private" / "owner-games"
DUMP = ART / "native-frames" / "vndb" / "dump" / "db"
NATIVE = {"vita": (960, 544), "psp": (480, 272)}
MIN_CORR = 0.95          # eval/tools/leakcheck.py's default: a near-duplicate of a protected screenshot
# The owner's drops: the layout read_samples reads, the archive, and the public table's file.
DROPS = {
    "20261013": {"layout": "samples", "archive": "vn-ocr-dataset-20261013.zip",
                 "public": "data/manifests/owner_drop_20261013.json"},
    "20261014": {"layout": "games", "archive": "VN-Screenshots-SFW-2026-10-14-5050.zip",
                 "public": "data/manifests/owner_drop_20261014.json"},
}

# The drop's folders, identified on VNDB's 2026-10-13 dump (title, platform, the Japanese release; the capture
# slot names the Vita title id). `release` is the Japanese console release; `release_vns` its other VNs.
GAMES = {
    ("psp", "Amagami"): {"vn": "v1281", "slug": "amagami", "title": "アマガミ (Amagami)",
                         "release": "r9959 エビコレ＋アマガミ (ULJS-00339)"},
    ("psp", "Corpse-Party-Blood-Covered"): {"vn": "v2632", "slug": "corpse-party", "title": "Corpse Party BloodCovered",
                                            "release": "r9521 (ULJM-05704)"},
    ("psp", "Grisaia-no-Kajitsu"): {"vn": "v5154", "slug": "grisaia-no-kajitsu", "title": "グリザイアの果実",
                                    "release": "r15389 (ULJM-06232)"},
    ("psp", "Narcissu-Portable"): {"vn": "v10", "slug": "narcissu-portable",
                                   "title": "ナルキッソス Portable (narcissu)", "release": "r7480 (ULJM-05674)",
                                   "release_vns": ["v70", "v1542", "v18736", "v18738"]},
    ("psp", "Planetarian"): {"vn": "v34", "slug": "planetarian", "title": "planetarian ～ちいさなほしのゆめ～",
                             "release": "r6626 (ULJM-05383)"},
    ("psp", "Sharin-no-Kuni"): {"vn": "v57", "slug": "sharin-no-kuni", "title": "車輪の国、向日葵の少女",
                                "release": "r4693 UMD-PG Edition or r17721 (ULJM-06022, with v84)"},
    ("vita", "CLANNAD"): {"vn": "v4", "slug": "clannad", "title": "CLANNAD",
                          "release": "r33730 (VLJM-35124, PCSG00415)"},
    ("vita", "Chaos-Child"): {"vn": "v14018", "slug": "chaos-child", "title": "CHAOS;CHILD",
                              "release": "r39462 (VLJM-35209, PCSG00555)"},
    ("vita", "Danganronpa-V3"): {"vn": "v18334", "slug": "danganronpa-v3", "title": "NEWダンガンロンパV3",
                                 "release": "r47935 (VLJS-05097, PCSG00936)"},
    ("vita", "Iwaihime-Matsuri"): {"vn": "v17863", "slug": "iwaihime-matsuri", "title": "祝姫 -祀-",
                                   "release": "r51223 (VLJM-38010, PCSG01083)"},
    ("vita", "Rewrite"): {"vn": "v751", "slug": "rewrite", "title": "Rewrite",
                          "release": "r27566 (VLJM-35125, PCSG00414)"},
}
# The 2026-10-14 drop's folders (games/<folder>), identified on VNDB's 2026-10-13 dump: `release` names the
# console title id of the capture slot and the VNDB release(s) it maps to (linked by title id where VNDB has
# the id, else the VN's Japanese releases on that platform); `release_vns` the release's other VNs.
GAMES_20261014 = {
    ("psp", "Amagami"): {"vn": "v1281", "slug": "amagami",
        "title": "アマガミ", "release": "r9959 (ULJS-00339); r10974 (ULJS-00349)"},
    ("psp", "Sharin-no-Kuni"): {"vn": "v57", "slug": "sharin-no-kuni",
        "title": "車輪の国、向日葵の少女",
        "release": "r4693 (PLCE-0001); r17721 (ULJM-06022) with v84", "release_vns": ["v84"]},
    ("psp", "Planetarian"): {"vn": "v34", "slug": "planetarian",
        "title": "planetarian ～ちいさなほしのゆめ～", "release": "r6626 (ULJM-05383); r13787 (ULJM-05383)"},
    ("psp", "Narcissu-Portable"): {"vn": "v10", "slug": "narcissu-portable",
        "title": "narcissu",
        "release": "r7480 (ULJM-05674) and r7145 (KAD-014), each with v70, v1542, v18736, v18738",
        "release_vns": ["v70", "v1542", "v18736", "v18738"]},
    ("vita", "Chaos-Child"): {"vn": "v14018", "slug": "chaos-child",
        "title": "CHAOS;CHILD", "release": "PCSG00555: r39462 (VLJM-35209); r49779 (FVGK-0128)"},
    ("vita", "Iwaihime-Matsuri"): {"vn": "v17863", "slug": "iwaihime-matsuri",
        "title": "祝姫", "release": "PCSG01083: r51223 (VLJM-38010)"},
    ("psp", "Danganronpa-1-PSP"): {"vn": "v7014", "slug": "danganronpa",
        "title": "ダンガンロンパ 希望の学園と絶望の高校生",
        "release": "NPJH50515: r21230 (NPJH-90164); r13888 (ULJS-00337)"},
    ("psp", "Second-Novel"): {"vn": "v4091", "slug": "second-novel",
        "title": "セカンドノベル ～彼女の夏、15分の記憶～", "release": "NPJH50279: r7967 (ULJS-00304); r7968"},
    ("psp", "Chaos-Head-NOAH"): {"vn": "v22505", "slug": "chaos-head-noah",
        "title": "CHAOS;HEAD NOAH", "release": "NPJH50551: r125603"},
    ("psp", "Hakuisei-Renai-Shoukougun-RE-Therapy"): {"vn": "v7301", "slug": "hakuisei-renai-re-therapy",
        "title": "白衣性恋愛症候群", "release": "NPJH00140: r14490 (CF00-20050); r14491 (ULJM-05935)"},
    ("psp", "No-Fate"): {"vn": "v3278", "slug": "no-fate",
        "title": "のーふぇいと！～only the power of will～",
        "release": "NPJH50360: r6399 (ALCH-00026); r6400 (ULJM-05610)"},
    ("psp", "Omoide-ni-Kawaru-Kimi-Memories-Off"): {"vn": "v1170", "slug": "omoide-ni-kawaru-kimi",
        "title": "想い出にかわる君 ～メモリーズオフ～", "release": "NPJH50129: r2221 (ULJM-05349); r10678 (FVGK-0009)"},
    ("psp", "Memories-Off-5-Togireta-Film"): {"vn": "v1325", "slug": "memories-off-5",
        "title": "Memories Off #5 とぎれたフィルム", "release": "NPJH50130: r2926 (FVGK-0014); r2927 (ULJM-05394)"},
    ("psp", "Memories-Off-Sorekara"): {"vn": "v1171", "slug": "memories-off-sorekara",
        "title": "Memories Off ～それから～", "release": "NPJH50132: r2423 (ULJM-05350); r2424 (FVGK-0010)"},
    ("vita", "Steins-Gate"): {"vn": "v2002", "slug": "steins-gate",
        "title": "STEINS;GATE", "release": "PCSG00147: r88732 (PCSG-00147)"},
    ("vita", "AMNESIA-V-Edition"): {"vn": "v7803", "slug": "amnesia-v-edition",
        "title": "AMNESIA", "release": "PCSG00203: r70114 (PCSG-00203)"},
    ("psp", "Memories-Off-6-T-Wave"): {"vn": "v1646", "slug": "memories-off-6",
        "title": "メモリーズオフ6 ～T-wave～", "release": "NPJH50382: r3225 (ULJM-05467); r3226 (FVGK-0027)"},
    ("psp", "Time-Travelers-PSP"): {"vn": "v10856", "slug": "time-travelers",
        "title": "タイムトラベラーズ", "release": "NPJH50597: r22079 (ULJM-06085)"},
    ("psp", "Solfege-Sweet-Harmony"): {"vn": "v2113", "slug": "solfege-sweet-harmony",
        "title": "ソルフェージュ", "release": "NPJH50095: r4133 (ULJM-05367); r11618 (GWS-0002)"},
    ("psp", "Trick-Logic"): {"vn": "v11713", "slug": "trick-logic",
        "title": "Trick x Logic Season 1", "release": "NPJG00025: r24291 (UCJS-10097)"},
    ("psp", "Da-Capo-Plus-Situation"): {"vn": "v264", "slug": "da-capo-plus-situation",
        "title": "D.C.～ダ・カーポ～", "release": "NPJH50731: r166234"},
    ("psp", "Dunamis15"): {"vn": "v7018", "slug": "dunamis15",
        "title": "DUNAMIS15", "release": "NPJH50641: r20183 (FVGK-0062); r21545 (ULJM-06119)"},
    ("vita", "Hakuisei-Aijou-Izonshou"): {"vn": "v16610", "slug": "hakuisei-aijou-izonshou",
        "title": "白衣性愛情依存症", "release": "PCSG00577: r38048; r45123"},
    ("psp", "Missingparts-Complete"): {"vn": "v16386", "slug": "missing-parts-complete",
        "title": "MISSING PARTS the TANTEI stories", "release": "NPJH50689: r37496"},
    ("vita", "Steins-Gate-Senkei-Kousoku-no-Phenogram"): {"vn": "v11660", "slug": "steins-gate-phenogram",
        "title": "STEINS;GATE 線形拘束のフェノグラム", "release": "PCSG00250: r28561 (VLJM-30061); r28562 (FVGK-0103)"},
    ("vita", "Steins-Gate-0"): {"vn": "v17102", "slug": "steins-gate-0",
        "title": "STEINS;GATE 0", "release": "PCSG00673: r40964 (VLJM-35259); r124594"},
    ("vita", "Steins-Gate-Hiyoku-Renri-no-Darling"): {"vn": "v6618", "slug": "steins-gate-darling",
        "title": "STEINS;GATE 比翼恋理のだーりん", "release": "PCSG00146: r55781 (PCSG-00146)"},
    ("vita", "Robotics-Notes-Elite"): {"vn": "v5883", "slug": "robotics-notes-elite",
        "title": "ROBOTICS;NOTES", "release": "PCSG00352: r32892; r32893"},
    ("vita", "Collar-x-Malice"): {"vn": "v18160", "slug": "collar-x-malice",
        "title": "Collar x Malice", "release": "PCSG00866: r45942 (PCSG-00866)"},
    ("vita", "Plastic-Memories"): {"vn": "v19441", "slug": "plastic-memories",
        "title": "プラスティック・メモリーズ", "release": "PCSG00931: r46361 (VLJM-30201); r95013"},
    ("vita", "Ikemen-Bakumatsu-Unmei-no-Koi-3-61-3-63"): {"vn": "v18777", "slug": "ikemen-bakumatsu",
        "title": "イケメン幕末◆運命の恋", "release": "PCSG01020: r49890"},
    ("psp", "Disgaea-Infinite"): {"vn": "v3333", "slug": "disgaea-infinite",
        "title": "ディスガイア インフィニット", "release": "NPJH00010: r6511 (NPJH-00010)"},
    ("vita", "VA-11-HALL-A"): {"vn": "v18872", "slug": "va-11-hall-a",
        "title": "VA-11 HALL-A", "release": "PCSG01072: r54063; r54496"},
    ("vita", "Side-Kicks"): {"vn": "v19743", "slug": "side-kicks",
        "title": "Side Kicks!", "release": "PCSG01001: r48560 (PCSG-01001)"},
    ("vita", "Azayaka-na-Irodori-no-Naka-de-Kimi-Rashiku-3-61-3-67"): {"vn": "v21899", "slug": "azayaka-na-irodori",
        "title": "あざやかな彩りの中で、君らしく", "release": "PCSG01175: r54733"},
    ("vita", "Mikagami-Sumika-no-Seifuku-Katsudou-3-61-3-67"): {"vn": "v22174", "slug": "mikagami-sumika",
        "title": "見鏡澄香の制服活動", "release": "PCSG01193: r55634"},
    ("vita", "Present-for-you"): {"vn": "v5386", "slug": "present-for-you",
        "title": "Present for you ～わたしをあ・げ・る～", "release": "PCSG00123: r25513 (PCSG-00123)"},
    ("vita", "Moujuutsukai-to-Oujisama"): {"vn": "v4540", "slug": "moujuutsukai-to-oujisama",
        "title": "猛獣使いと王子様", "release": "PCSG00244: r70291 (PCSG-00244)"},
    ("vita", "Shin-Kamaitachi-no-Yoru-11-Ninme-no-Suspect"): {"vn": "v10561", "slug": "shin-kamaitachi-no-yoru",
        "title": "真かまいたちの夜 11人目の訪問者（サスペクト）",
        "release": "PCSG00010: r21357 (VLJS-00006); r123647 (VLJS-50002)"},
    ("vita", "Princess-Arthur"): {"vn": "v11108", "slug": "princess-arthur",
        "title": "Princess Arthur", "release": "PCSG00271: r70290 (PCSG-00271)"},
    ("vita", "Binary-Star"): {"vn": "v14787", "slug": "binary-star",
        "title": "Binary Star", "release": "PCSG00389: r33986 (PCSG-00389)"},
    ("vita", "Reine-des-Fleurs"): {"vn": "v14919", "slug": "reine-des-fleurs",
        "title": "レンドフルール", "release": "PCSG00405: r33247 (PCSG-00405)"},
    ("vita", "Hyakka-Yakou"): {"vn": "v15773", "slug": "hyakka-yakou",
        "title": "百華夜光", "release": "PCSG00477: r35966 (PCSG-00477)"},
    ("vita", "Chouchou-Jiken-Rhapsodic"): {"vn": "v19851", "slug": "chouchou-jiken-rhapsodic",
        "title": "蝶々事件ラブソディック", "release": "PCSG01066: r52575 (PCSG-01066)"},
    ("vita", "Otoko-Yuukaku"): {"vn": "v15231", "slug": "otoko-yuukaku",
        "title": "逆転 吉原", "release": "PCSG00487: r37386 (PCSG-00487)"},
    ("vita", "Yahari-Game-demo-Ore-no-Seishun-Love-Come-wa-Machigatteiru"): {"vn": "v12630", "slug": "oregairu",
        "title": "やはりゲームでも俺の青春ラブコメはまちがっている。",
        "release": "PCSG00215: r26993 (VLJM-30052); r48631 (FVGK-0090)"},
    ("vita", "Sweet-Clown"): {"vn": "v16460", "slug": "sweet-clown",
        "title": "SWEET CLOWN ~午前三時のオカシな道化師~", "release": "PCSG00528: r50738 (PCSG-00528)"},
    ("vita", "Hanasaku-Manimani"): {"vn": "v11677", "slug": "hanasaku-manimani",
        "title": "花咲くまにまに", "release": "PCSG00366: r35839 (PCSG-00366)"},
    ("psp", "Juusei-to-Diamond"): {"vn": "v14909", "slug": "juusei-to-diamond",
        "title": "銃声とダイヤモンド", "release": "NPJG00016: r129019 (NPJG-90040); r33053"},
    ("psp", "Hayarigami-2-Portable-Keishichou-Kaii-Jiken-File"): {"vn": "v21259", "slug": "hayarigami-2-portable",
        "title": "流行り神2 警視庁怪異事件ファイル", "release": "NPJH50002: r51746; r51751"},
    ("psp", "Princess-Maker-4-Portable"): {"vn": "v727", "slug": "princess-maker-4-portable",
        "title": "プリンセスメーカー4", "release": "NPJH50069: r114338 (NPJH-50069)"},
    ("psp", "Misshitsu-no-Sacrifice"): {"vn": "v4185", "slug": "misshitsu-no-sacrifice",
        "title": "密室のサクリファイス", "release": "NPJH50224: r8096; r8097 (ULJS-00251)"},
    ("psp", "Fate-Extra-CCC"): {"vn": "v12369", "slug": "fate-extra-ccc",
        "title": "Fate/EXTRA CCC", "release": "NPJH50505: r55779; r117809 (ULJS-19089)"},
    ("psp", "Fate-Extra"): {"vn": "v7602", "slug": "fate-extra",
        "title": "フェイト/エクストラ", "release": "NPJH50247: r15247 (ULJS-00254); r24358 (ULJS-00253)"},
    ("psp", "Princess-Maker-5-Portable"): {"vn": "v728", "slug": "princess-maker-5-portable",
        "title": "プリンセスメーカー5", "release": "NPJH50070: r114349 (NPJH-50070)"},
    ("psp", "Da-Capo-II-Plus-Situation"): {"vn": "v266", "slug": "da-capo-ii-plus-situation",
        "title": "D.C.II ～ダ・カーポII～",
        "release": "NPJH50732: r9781 (ULJM-05718~9) with v264; r10620 (KAD-015) with v264"},
    ("psp", "MagusTale-Eternity"): {"vn": "v689", "slug": "magustale-eternity",
        "title": "MagusTale ～世界樹と恋する魔法使い～", "release": "NPJH50835: r7521 (P-00013); r7522 (ULJM-05577)"},
    ("psp", "Custom-Drive"): {"vn": "v10623", "slug": "custom-drive",
        "title": "カスタムドライブ", "release": "NPJH50649: r34123 (NPJH-50649)"},
    ("psp", "Utawarerumono-Portable"): {"vn": "v3", "slug": "utawarerumono-portable",
        "title": "うたわれるもの", "release": "NPJH50882: r117205 (NPJH-50882)"},
    ("psp", "BlazBlue-Continuum-Shift-Extend"): {"vn": "v6572", "slug": "blazblue-cs-extend",
        "title": "BLAZBLUE CONTINUUM SHIFT", "release": "NPJH50582: r15206 (ULJM-05850); r21483 (ULJM-06061)"},
    ("psp", "Pastel-Chime-Continue"): {"vn": "v1080", "slug": "pastel-chime-continue",
        "title": "ぱすてるチャイムContinue", "release": "NPJH50550: r11532 (ULJM-05788); r11533 (FVGK-0045)"},
    ("vita", "Zero-Escape-Virtue-s-Last-Reward"): {"vn": "v7809", "slug": "zero-escape-vlr",
        "title": "極限脱出ADV 善人シボウデス", "release": "PCSG00020: r15652 (VLJS-00009); r19906"},
    ("vita", "Sakura-Sou-no-Pet-na-Kanojo"): {"vn": "v11173", "slug": "sakura-sou-no-pet-na-kanojo",
        "title": "さくら荘のペットな彼女", "release": "PCSG00129: r22795 (VLJS-05017); r24450 (VLJS-05016)"},
    ("vita", "XBlaze-Code-Embryo"): {"vn": "v13059", "slug": "xblaze-code-embryo",
        "title": "XBLAZE CODE:EMBRYO", "release": "PCSG00136: r50886 (VLJS-00047)"},
    ("vita", "Cure-Mate-Club"): {"vn": "v2923", "slug": "cure-mate-club",
        "title": "Cure Mate Club", "release": "PCSG00169: r25515"},
    ("vita", "Hotchkiss"): {"vn": "v7967", "slug": "hotchkiss", "title": "ホチキス", "release": "PCSG00200: r48635"},
    ("vita", "bitter-smile"): {"vn": "v2954", "slug": "bitter-smile",
        "title": "Bitter Smile.", "release": "PCSG00242: r28766 (TGCS-00003); r28767 (VLJM-30059)"},
    ("vita", "Infinite-Stratos-2-Ignition-Hearts"): {"vn": "v13865", "slug": "infinite-stratos-2",
        "title": "IS＜インフィニット・ストラトス＞2 イグニッション・ハーツ",
        "release": "PCSG00270: r30207 (FVGK-0102); r38969 (VLJM-30065)"},
    ("vita", "KissBell"): {"vn": "v10641", "slug": "kissbell", "title": "キスベル", "release": "PCSG00413: r34134"},
    ("vita", "CHAOS-HEAD-Love-Chu-Chu"): {"vn": "v3091", "slug": "chaos-head-love-chu-chu",
        "title": "カオスヘッド らぶChu☆Chu!",
        "release": "PCSG00435: r38137 (VLJM-35126) with v22505; r38138 (FVGK-0117) with v22505"},
    ("psp", "Tenshin-Ranman-Happy-Go-Lucky"): {"vn": "v1322", "slug": "tenshin-ranman",
        "title": "天神乱漫 LUCKY or UNLUCKY!?", "release": "NPJH50297: r5724 (ULJM-05634); r73326 (ULJM-05633)"},
    ("vita", "Teikoku-Kaigun-Koi-Bojou"): {"vn": "v17285", "slug": "teikoku-kaigun-koi-bojou",
        "title": "帝国海軍恋慕情 ～明治横須賀行進曲～", "release": "PCSG00547: r39996 (PCSG-00547)"},
    ("vita", "Ray-Gigant"): {"vn": "v24957", "slug": "ray-gigant",
        "title": "レイギガント", "release": "PCSG00552: r61259 (VLJS-00111); r97382"},
    ("vita", "Cross-Ange-Tenshi-to-Ryuu-no-Rondo-tr"): {"vn": "v19100", "slug": "cross-ange",
        "title": "クロスアンジュ 天使と竜の輪舞tr.", "release": "PCSG00597: r45366 (VLJS-00110)"},
    ("psp", "Hayarigami-Portable-Keishichou-Kaii-Jiken-File"): {"vn": "v2063", "slug": "hayarigami-portable",
        "title": "流行り神 警視庁怪異事件ファイル", "release": "NPJH50001: r43386 (ULJS-00035); r51750"},
    ("psp", "Tantei-Jinguuji-Saburo-Hai-to-Diamond"): {"vn": "v19010", "slug": "jinguuji-saburo-hai-to-diamond",
        "title": "探偵 神宮寺三郎 灰とダイヤモンド", "release": "NPJH50079: r8815 (ULJM-05529)"},
    ("psp", "Suzukaze-no-Melt"): {"vn": "v3992", "slug": "suzukaze-no-melt",
        "title": "涼風のメルト -Where wishes are drawn to each other-",
        "release": "NPJH50801: r21074 (P-00039); r21895 (ULJM-06053)"},
    ("vita", "Anoko-wa-Ore-Kara-Hanarenai"): {"vn": "v10271", "slug": "anoko-wa-ore-kara-hanarenai",
        "title": "彼女はオレからはなれない", "release": "PCSG00342: r31028 (TGCS-00004); r31029 (VLJM-30078)"},
    ("vita", "Vamwolf-Cross-Limited-Edition"): {"vn": "v16468", "slug": "vamwolf-cross",
        "title": "Vamwolf Cross†", "release": "PCSG00467: r37753 (PCSG-00467)"},
    ("vita", "Ozmafia-Vivace"): {"vn": "v7833", "slug": "ozmafia-vivace",
        "title": "OZMAFIA!!", "release": "PCSG00496: r70035 (PCSG-00496)"},
    ("vita", "Parfait"): {"vn": "v683", "slug": "parfait",
        "title": "パルフェ ～ショコラ second brew～", "release": "PCSG00549: r48636"},
    ("vita", "5-nin-no-Koi-Prince"): {"vn": "v16607", "slug": "5-nin-no-koi-prince",
        "title": "5人の恋プリンス ~ヒミツの契約結婚~", "release": "PCSG00590: r38581 (PCSG-00590)"},
    ("vita", "Abunai-Koi-no-Sousashitsu-Eternal-Happiness"): {"vn": "v16467", "slug": "abunai-koi-no-sousashitsu",
        "title": "アブナイ恋の捜査室 ~Eternal Happiness~", "release": "PCSG00591: r39882 (PCSG-00591)"},
    ("vita", "Fate-Extella-3-61-3-61"): {"vn": "v19119", "slug": "fate-extella",
        "title": "Fate/EXTELLA", "release": "PCSG00600: r45447 (VLJM-30158); r51550 (PLJM-80176)"},
    ("vita", "Tokyo-Ghoul-Jail"): {"vn": "v17936", "slug": "tokyo-ghoul-jail",
        "title": "東京喰種 JAIL", "release": "PCSG00615: r41299 (VLJS-00113); r58179"},
    ("vita", "PCSG00641"): {"vn": "v17902", "slug": "kurogane-kaikitan",
        "title": "クロガネ回姫譚 -一期一会-", "release": "PCSG00641: r41222 (PCSG-00641)"},
    ("vita", "Tsuki-ni-Yori-Sou-Otome-no-Sahou"): {"vn": "v10680", "slug": "tsuki-ni-yorisou-otome-no-sahou",
        "title": "月に寄りそう乙女の作法", "release": "PCSG00648: r40209 (VLJM-35249)"},
}
# Owner decisions that override the VNDB check pattern. They never override an eval or development set.
OWNER_DECISIONS = {
    "v34": {"role": "training", "by": "owner 2026-10-13 19:21Z and 20:47Z",
            "note": "planetarian (Key, all releases) is training data, allowed even where it overlaps an eval set "
                    "(realplus/shots.OWNER_ALLOWED). Without that exemption reserved_vns reserves it only as "
                    "shared-release (the Key 10th Memorial BOX, r3748, a PC compilation with CLANNAD and Tomoyo "
                    "After); it has no rows in vita-test, psp-test, teacher_val or devnative. Its PSP port and "
                    "CLANNAD's Vita port are both Key/Prototype releases: a CLANNAD score of any model trained on "
                    "these frames must disclose that exposure."},
}
CLANNAD_REASONS = {"clannad", "clannad-family", "clannad-relation"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_facts(path):
    """(width, height, PIL-RGB sha256, devnative pixel sha256) of one image."""
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        a = np.ascontiguousarray(np.asarray(im.convert("RGB"), dtype=np.uint8))
    h = hashlib.sha256(f"{a.shape[0]}x{a.shape[1]}x3".encode())
    h.update(a.tobytes())
    return a.shape[1], a.shape[0], hashlib.sha256(a.tobytes()).hexdigest(), h.hexdigest()


def source_resolution(platform, w, h):
    return "native" if NATIVE.get(platform) == (w, h) else f"non-native {w}x{h}"


def read_samples(drop):
    """The drop's full frames as rows {image, platform, title, collection, sequence, sha256, dimensions, ...}."""
    if layout_of(drop) == "games":
        return read_games(drop)
    rows = [json.loads(x) for x in (Path(drop) / "samples.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    for r in rows:
        p = Path(r["image"])
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"unsafe sample path {r['image']}")
    return rows


def layout_of(drop):
    """'samples' (2026-10-13: samples.jsonl) or 'games' (2026-10-14: dataset-index.json + games/<folder>/)."""
    drop = Path(drop)
    if (drop / "samples.jsonl").exists():
        return "samples"
    if (drop / "dataset-index.json").exists() and (drop / "games").is_dir():
        return "games"
    raise ValueError(f"{drop}: neither samples.jsonl nor dataset-index.json + games/")


def drop_tag(drop):
    """The DROPS key of a drop directory (its layout's drop)."""
    lay = layout_of(drop)
    return next(k for k, v in DROPS.items() if v["layout"] == lay)


def read_games(drop):
    """The 2026-10-14 layout: one row per frame of every games/<folder>/manifest.jsonl, the platform from the capture
    slot (vita-pcsg00147, psp-npjh50515, psp-amagami), the category from the row (dialogue / title_screen /
    settings_screen; a reviewed new-dialogue frame without one is dialogue)."""
    drop = Path(drop)
    index = json.loads((drop / "dataset-index.json").read_text(encoding="utf-8"))
    rows = []
    for t in index["titles"]:
        folder = Path(t["folder"])
        if folder.is_absolute() or ".." in folder.parts or folder.parts[0] != "games":
            raise ValueError(f"unsafe folder {t['folder']}")
        platform = t["source_slot"].split("-", 1)[0]
        if platform not in NATIVE:
            raise ValueError(f"{t['folder']}: unknown platform in slot {t['source_slot']}")
        for x in (drop / folder / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
            if not x.strip():
                continue
            r = json.loads(x)
            p = Path(r["image"])
            if p.is_absolute() or ".." in p.parts:
                raise ValueError(f"unsafe image path {r['image']}")
            cat = r.get("category") or ("dialogue" if r.get("visually_reviewed_new_dialogue") else "unknown")
            rows.append({"image": str(folder / p), "platform": platform, "title": folder.name,
                         "collection": r["source_run"], "sequence": r.get("sequence"), "sha256": r["sha256"],
                         "dimensions": r["dimensions"], "category": cat, "slot": t["source_slot"],
                         "drop_vns": list(t.get("canonical_vndb_ids") or [])})
    return rows


# ----------------------------------------------------------------------------- the sets a VN may sit in

def npz_vns(path):
    """Counter(vn -> rows) of a line pool's meta table (an npz without one holds no lines: empty)."""
    import numpy as np
    z = np.load(path)
    if "meta" not in z.files:
        return collections.Counter()
    return collections.Counter(m.get("vn") for m in json.loads(str(z["meta"])))


def set_rows(root=ROOT, art=ART):
    """{set name: Counter(vn -> rows)} of every evaluation and development set on disk."""
    out = {}
    for s in ("vita-test", "psp-test"):
        p = root / "eval" / "manifests" / f"{s}.jsonl"
        if p.exists():
            out[s] = collections.Counter(json.loads(x)["vn"] for x in p.read_text(encoding="utf-8").splitlines()
                                         if x.strip())
    p = art / "teacher" / "teacher_val.npz"
    if p.exists():
        out["teacher_val"] = npz_vns(p)
    dn = collections.Counter()
    for p in sorted((art / "devnative").rglob("*.npz")) if (art / "devnative").exists() else ():
        dn += npz_vns(p)
    for p in sorted((root / "devnative" / "manifests").glob("*.jsonl")):
        dn += collections.Counter(json.loads(x).get("vn") for x in p.read_text(encoding="utf-8").splitlines()
                                  if x.strip())
    out["devnative"] = dn
    return out


def eval_membership(vn, sets, test_all, dev, fresh):
    """The eval / development sets a VN is in (empty: none)."""
    hit = {k: c[vn] for k, c in sets.items() if c.get(vn)}
    if vn in test_all:
        hit["test_games_all"] = True
    if is_test_vn(vn):
        hit["is_test_vn"] = True
    if vn in dev:
        hit["development_vns"] = True
    if vn in fresh:
        hit["fresh_native"] = True
    return hit


# The sets that make a game test-only (docs/reservations.md); the other memberships eval_membership reports
# (test_games_all, is_test_vn, development_vns, fresh_native, teacher_val, devnative) are information only.
EVAL_KEYS = ("vita-test", "psp-test")


def decide(vn, reason, sets_hit, owner=OWNER_DECISIONS, leak_frames=0):
    """(role, why) of one game; role is training, test or blocked."""
    if vn in SH.CLANNAD or vn in SH.CLANNAD_FAMILY or reason in CLANNAD_REASONS:
        return "test", f"protected: CLANNAD ({reason or 'clannad'}), test only (owner)"
    in_eval = sorted(k for k in sets_hit if k in EVAL_KEYS)
    if vn in owner:
        o = owner[vn]
        if in_eval:
            return "blocked", f"{o['by']} says {o['role']}, but the VN is in {', '.join(in_eval)}: stopped"
        role, why = o["role"], f"{o['by']}: {o['role']} (reserved_vns: {reason or 'not reserved'})"
    elif reason and reason.startswith("dev-sample"):
        role, why = "training", (f"reserved: {reason}, but owner screenshots train (owner 2026-10-14 13:25Z): "
                                 "re-run realplus/shots.py reserve so the development sample leaves this game out")
    elif reason:
        role, why = "test", f"reserved: {reason}"
    elif in_eval:
        role, why = "test", f"in {', '.join(in_eval)}"
    else:
        role, why = "training", "not reserved, in no eval or development set"
    if role == "training" and leak_frames:
        return "blocked", f"{leak_frames} frame(s) at or above {MIN_CORR} against a protected screenshot"
    return role, why


def folder_name(platform, slug, vn, role, existing=(), any_suffix=False):
    """<platform>-<slug>-v<id>[-TEST]; an existing folder of the same VN and role wins. any_suffix (a later drop,
    under the 2026-10-14 policy where the -TEST suffix no longer decides a folder's role, owner_frames.folder_info):
    failing that, any existing folder of the same VN wins, so a VN keeps one folder."""
    test = role != "training"
    pat = re.compile(rf"-{re.escape(vn)}{'-TEST' if test else ''}$")
    same = sorted(n for n in existing if pat.search(n))
    if not same and any_suffix:
        same = sorted(n for n in existing if re.search(rf"-{re.escape(vn)}(?:-TEST)?$", n))
    return same[0] if same else f"{platform}-{slug}-{vn}{'-TEST' if test else ''}"


# ----------------------------------------------------------------------------- leak check (thumbnails)

def protected_shots(art=ART):
    """{group: [paths]}: eval's candidate screenshots (every test screenshot is one), devnative's cohort,
    teacher_val's screenshots."""
    dn = art / "devnative" / "shots"
    out = {"eval-vita": sorted((art / "eval" / "vita" / "shots").glob("*.jpg")),
           "eval-psp": sorted((art / "eval" / "psp" / "shots").glob("*.jpg")),
           "devnative": sorted(p for p in dn.rglob("*") if p.suffix.lower() in (".jpg", ".png")) if dn.exists() else []}
    tv = art / "teacher" / "teacher_val.npz"
    if tv.exists():
        import numpy as np
        meta = json.loads(str(np.load(tv)["meta"]))
        paths = {art / "realdata" / "shots" / m["kind"] / ("_".join(m["id"].split("_")[:2]) + ".jpg") for m in meta}
        out["teacher_val"] = sorted(p for p in paths if p.exists())
    return {k: v for k, v in out.items() if v}


def leak_scores(frames, groups):
    """Per frame index: {group: best thumbnail correlation}."""
    import numpy as np
    sys.path.insert(0, str(ROOT / "eval" / "tools"))
    sys.path.insert(0, str(ROOT / "eval"))
    from leakcheck import thumb
    T = np.stack([thumb(p) for p in frames])
    out = [dict() for _ in frames]
    for g, paths in groups.items():
        M = np.stack([thumb(p) for p in paths])
        best = (T @ M.T).max(axis=1)
        for i, b in enumerate(best):
            out[i][g] = round(float(b), 4)
    return out


IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FOLDER_VN = re.compile(r"-(v\d+)(?:-TEST)?$")


def test_set_shots(art=ART, root=ROOT, owner_games=OWNER_GAMES, test=None):
    """{group: [paths]} of the three test sets only (docs/reservations.md): vita-test's and psp-test's screenshots
    (eval's local copies of every manifest row's screenshot; a missing one is an error) and every frame of an
    owner-games folder whose VN is a CLANNAD-family VN of the test reservation (the CLANNAD long run)."""
    from common import test_set_vns
    sys.path.insert(0, str(root / "eval"))
    from fetch import shot_name
    test = test_set_vns() if test is None else test
    out = {}
    for s, plat in (("vita-test", "vita"), ("psp-test", "psp")):
        rows = [json.loads(x) for x in (root / "eval" / "manifests" / f"{s}.jsonl").read_text(encoding="utf-8")
                .splitlines() if x.strip()]
        paths = sorted({art / "eval" / plat / "shots" / shot_name(r["vn"], r["url"]) for r in rows})
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{s}: {len(missing)} screenshots not on disk, e.g. {missing[0]}")
        out[s] = paths
    clannad = {v for v, r in test.items() if r.startswith("clannad")}
    cl = []
    for d in sorted(Path(owner_games).iterdir()) if Path(owner_games).exists() else ():
        m = FOLDER_VN.search(d.name)
        if d.is_dir() and m and m.group(1) in clannad:
            cl += sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT)
    if cl:
        out["clannad"] = cl
    return out


def owner_pixels(owner_games=OWNER_GAMES):
    """{decoded-pixel sha256: "folder/file"} of every frame already in owner-games (PIL's RGB hash and devnative's,
    first file in name order wins): what a later drop's frames are deduplicated against."""
    out = {}
    for d in sorted(Path(owner_games).iterdir()) if Path(owner_games).exists() else ():
        if not d.is_dir() or d.name.startswith("_"):
            continue
        for p in sorted(x for x in d.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXT):
            _, _, rgb, dn = frame_facts(p)
            out.setdefault(rgb, f"{d.name}/{p.name}")
            out.setdefault(dn, f"{d.name}/{p.name}")
    return out


# ----------------------------------------------------------------------------- check

def check(drop, dump=DUMP, leak=False, owner_games=OWNER_GAMES, why=None, sets=None, why_strict=None, known=None,
          groups=None):
    drop = Path(drop)
    lay, tag = layout_of(drop), drop_tag(drop)
    table = GAMES if lay == "samples" else GAMES_20261014
    samples = read_samples(drop)
    if why is None:
        d = SH.load_dump(dump)
        why = SH.reserved_vns(d)
        allowed, SH.OWNER_ALLOWED = SH.OWNER_ALLOWED, {}
        try:
            why_strict = SH.reserved_vns(d)                 # what the owner's exemptions lifted
        finally:
            SH.OWNER_ALLOWED = allowed
    why_strict = why if why_strict is None else why_strict
    test_all = set(SH.test_games()["all"])
    dev, fresh = SH.dev_vns(), SH.fresh_native_vns()
    sets = set_rows() if sets is None else sets
    existing = sorted(p.name for p in Path(owner_games).iterdir() if p.is_dir()) if Path(owner_games).exists() else []
    frames = []
    for r in samples:
        w, h, rgb, dn = frame_facts(drop / r["image"])
        if sha256_file(drop / r["image"]) != r["sha256"] or [w, h] != list(r["dimensions"]):
            raise ValueError(f"{r['image']}: hash or size differs from the drop's manifest")
        f = {"image": r["image"], "platform": r["platform"], "title": r["title"],
             "collection": r["collection"], "sequence": r.get("sequence"), "sha256": r["sha256"],
             "pixel_sha256": rgb, "dn_pixel_sha256": dn, "dims": [w, h],
             "source_resolution": source_resolution(r["platform"], w, h)}
        if lay == "games":
            f.update(category=r["category"], slot=r["slot"])
        frames.append(f)
    if lay == "games":
        # pixel copies of a frame already in owner-games (the first drop) or of an earlier frame of this drop
        known = owner_pixels(owner_games) if known is None else known
        seen = {}
        for f in frames:
            dup = known.get(f["pixel_sha256"]) or known.get(f["dn_pixel_sha256"]) or seen.get(f["pixel_sha256"])
            if dup:
                f["duplicate_of"] = dup
            else:
                seen[f["pixel_sha256"]] = f"drop:{f['image']}"
    if leak and lay == "samples":
        for f, s in zip(frames, leak_scores([drop / f["image"] for f in frames], protected_shots())):
            f["leak"] = s
    by_game = collections.defaultdict(list)
    for f in frames:
        by_game[(f["platform"], f["title"])].append(f)
    games = []
    for key in sorted(set(table) | set(by_game)):
        g = table.get(key)
        if g is None:
            raise ValueError(f"unidentified game folder {key}: add it to GAMES")
        fs = by_game.get(key, [])
        vn = g["vn"]
        hit = eval_membership(vn, sets, test_all, dev, fresh)
        n_leak = sum(1 for f in fs if max((f.get("leak") or {"-": 0}).values()) >= MIN_CORR)
        role, reason = decide(vn, why.get(vn), hit, leak_frames=n_leak)
        dims = collections.Counter(tuple(f["dims"]) for f in fs)
        row = {"platform": key[0], "drop_title": key[1], "vn": vn, "title": g["title"], "release": g["release"],
               "release_vns": g.get("release_vns", []),
               "release_vns_reserved": {v: why.get(v) for v in g.get("release_vns", []) if why.get(v)},
               "frames": len(fs), "dims": {f"{a}x{b}": n for (a, b), n in sorted(dims.items())},
               "source_resolution": sorted({f["source_resolution"] for f in fs}),
               "collections": dict(collections.Counter(f["collection"] for f in fs)),
               "reserved_vns": why.get(vn), "reserved_without_owner_allowed": why_strict.get(vn),
               "sets": hit, "role": role, "why": reason,
               "owner_decision": OWNER_DECISIONS.get(vn), "leak_frames": n_leak,
               "leak_max": {k: max(f["leak"][k] for f in fs) for k in (fs[0].get("leak") or {})} if fs else {},
               "folder": folder_name(key[0], g["slug"], vn, role, existing, any_suffix=lay == "games") if fs else None}
        if lay == "games":
            row.update(categories=dict(collections.Counter(f["category"] for f in fs)),
                       duplicates=sum("duplicate_of" in f for f in fs),
                       duplicates_of_owner_games=sum(not str(f.get("duplicate_of", "drop:")).startswith("drop:")
                                                     for f in fs),
                       slot=fs[0]["slot"] if fs else None)
        games.append(row)
    if lay == "games":
        role = {(g["platform"], g["drop_title"]): g["role"] for g in games}
        train = [f for f in frames if role[(f["platform"], f["title"])] == "training"]
        if leak and train:
            # the three test sets only, plus this drop's own frames of test games
            groups = test_set_shots(owner_games=owner_games) if groups is None else dict(groups)
            drop_test = [drop / f["image"] for f in frames if role[(f["platform"], f["title"])] != "training"]
            if drop_test:
                groups["drop-test"] = drop_test
            for f, s in zip(train, leak_scores([drop / f["image"] for f in train], groups)):
                f["leak"] = s
                f["leak_dropped"] = max(s.values()) >= MIN_CORR
        for g in games:
            fs = by_game.get((g["platform"], g["drop_title"]), [])
            g["leak_frames"] = sum(bool(f.get("leak_dropped")) for f in fs)
            g["leak_max"] = {k: max(f["leak"][k] for f in fs) for k in (fs[0].get("leak") or {})} if fs else {}
            g["new_frames"] = sum("duplicate_of" not in f and not f.get("leak_dropped") for f in fs)
    return {"kind": "owner-drop-plan", "schema": 1, "drop": str(drop), "drop_tag": tag, "layout": lay,
            "frames": frames, "games": games,
            "computed": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


# ----------------------------------------------------------------------------- sort

def dest_name(f, tag=None):
    """The frame's file name in its game folder: the long-run CLANNAD frames keep inputs/'s name
    (NNNNNN-native.png), the first drop's current-run frames their six-digit sequence name; a later drop's frames
    get d<date>_ in front (d20261014_000001.png), next to the first drop's in a shared folder."""
    name = Path(f["image"]).name
    return name if tag in (None, "20261013") else f"d{tag}_{name}"


def sort(plan, dest=OWNER_GAMES, link=True):
    """Hardlink (or copy) every frame into its game folder; writes the manifest and the ingest decision there
    (manifest.jsonl / ingest.json for the first drop, manifest_<date>.jsonl / ingest_<date>.json for a later one).
    A later drop's duplicate and leak-dropped frames are listed in the manifest but not linked."""
    import shutil
    drop, dest = Path(plan["drop"]), Path(dest)
    tag = plan.get("drop_tag", "20261013")
    later = tag != "20261013"
    games = {(g["platform"], g["drop_title"]): g for g in plan["games"]}
    done = collections.Counter()
    rows = collections.defaultdict(list)
    for f in plan["frames"]:
        g = games[(f["platform"], f["title"])]
        out = dest / g["folder"]
        out.mkdir(parents=True, exist_ok=True)
        target = out / dest_name(f, tag if later else None)
        skip = "duplicate" if f.get("duplicate_of") else "leak" if f.get("leak_dropped") else None
        if skip is None:
            if target.exists():
                if sha256_file(target) != f["sha256"]:
                    raise FileExistsError(f"{target} exists with other bytes; never overwritten")
            elif link:
                os.link(drop / f["image"], target)
            else:
                shutil.copy2(drop / f["image"], target)
            done[g["folder"]] += 1
        row = {"file": target.name if skip is None else None, "drop_image": f["image"], "sha256": f["sha256"],
               "pixel_sha256": f["pixel_sha256"], "dn_pixel_sha256": f["dn_pixel_sha256"],
               "dims": f["dims"], "source_resolution": f["source_resolution"],
               "platform": f["platform"], "vn": g["vn"], "collection": f["collection"],
               "sequence": f.get("sequence")}
        if later:
            row.update(category=f.get("category"), slot=f.get("slot"), skipped=skip,
                       duplicate_of=f.get("duplicate_of"), leak_max=max((f.get("leak") or {"-": None}).values()))
        rows[g["folder"]].append(row)
    for folder, rs in rows.items():
        g = next(x for x in plan["games"] if x["folder"] == folder)
        names = [r["file"] for r in rs if r["file"]]
        if len(set(names)) != len(names):
            raise ValueError(f"{folder}: two frames share a file name")
        man, ing = ("manifest.jsonl", "ingest.json") if not later else (f"manifest_{tag}.jsonl", f"ingest_{tag}.json")
        (dest / folder / man).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs),
                                         encoding="utf-8")
        info = {k: g[k] for k in ("platform", "vn", "title", "release", "role", "why", "reserved_vns", "sets",
                                  "frames", "dims", "source_resolution", "owner_decision", "leak_max")}
        arc = DROPS.get(tag, {}).get("archive", "")
        info.update(source=f"owner Google Drive drop {tag[:4]}-{tag[4:6]}-{tag[6:]} ({arc})",
                    private="copyrighted frames: never commit, never send to Codex or any third party")
        if later:
            info.update({k: g.get(k) for k in ("categories", "duplicates", "duplicates_of_owner_games", "leak_frames",
                                                "new_frames", "slot")})
        (dest / folder / ing).write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return dict(done)


# ----------------------------------------------------------------------------- public table

def public(plan, archive=None):
    keep = ("platform", "drop_title", "vn", "title", "release", "frames", "dims", "source_resolution", "collections",
            "reserved_vns", "reserved_without_owner_allowed", "release_vns_reserved", "sets", "role", "why",
            "leak_max", "leak_frames", "folder")
    later = ("slot", "categories", "duplicates", "duplicates_of_owner_games", "new_frames")
    games = [{k: g[k] for k in keep} | {k: g[k] for k in later if k in g} for g in plan["games"]]
    out = {"kind": "owner-drop-summary", "schema": 1, "archive": archive or {},
           "frames": sum(g["frames"] for g in plan["games"]),
           "by_role": dict(collections.Counter(g["role"] for g in plan["games"] for _ in range(g["frames"]))),
           "games": games, "owner_decisions": OWNER_DECISIONS, "min_corr": MIN_CORR,
           "note": "counts, ids and reasons only: no frame, no text, no per-frame hash"}
    if any("new_frames" in g for g in plan["games"]):
        out["new_frames_by_role"] = dict(collections.Counter(
            g["role"] for g in plan["games"] for _ in range(g.get("new_frames", 0))))
        out["duplicates"] = sum(g.get("duplicates", 0) for g in plan["games"])
        out["leak_frames"] = sum(g.get("leak_frames", 0) for g in plan["games"])
    blob = json.dumps(out, ensure_ascii=False)
    if '"image"' in blob or ".png" in blob:
        raise ValueError("public table names a frame")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("--drop", type=Path, required=True)
    p.add_argument("--dump", type=Path, default=DUMP)
    p.add_argument("--leak", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("sort")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--dest", type=Path, default=OWNER_GAMES)
    p.add_argument("--copy", action="store_true", help="copy instead of hardlinking")
    p = sub.add_parser("public")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--archive-sha256")
    p.add_argument("--archive-bytes", type=int)
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "check":
        plan = check(a.drop, a.dump, a.leak)
        a.out.write_text(json.dumps(plan, ensure_ascii=False) + "\n", encoding="utf-8")
        for g in plan["games"]:
            print(f"{g['platform']:4} {g['drop_title']:28} {g['vn']:7} {g['frames']:5} {g['role']:8} {g['why']}")
    elif a.cmd == "sort":
        plan = json.loads(a.plan.read_text(encoding="utf-8"))
        print(json.dumps(sort(plan, a.dest, link=not a.copy), indent=1))
    else:
        plan = json.loads(a.plan.read_text(encoding="utf-8"))
        name = DROPS[plan.get("drop_tag", "20261013")]["archive"]
        arc = {"name": name, "sha256": a.archive_sha256, "bytes": a.archive_bytes}
        a.out.write_text(json.dumps(public(plan, arc), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
