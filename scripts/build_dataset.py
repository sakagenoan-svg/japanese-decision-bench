"""Build data/*.jsonl from the authored examples in this file.

    uv run python scripts/build_dataset.py

Authored examples (T1, T2, T4, T5-clean) are written inline below. Derived examples are generated
deterministically:
  T3 katakana_halfwidth  = halfwidth(katakana(hiragana variant))  (character mapping, no model)
  T3 other variants      = hand-written rewrites (stored below)
  T5 medium / heavy      = clean text wrapped in label-neutral distractor text chosen with a fixed seed

All texts are synthetic. The setting is a fictional peer-to-peer goods exchange service. No real users,
names, transaction IDs, or customer messages.
"""

from __future__ import annotations

import random
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.config import DATA_DIR, SEED  # noqa: E402
from bench.dataset import write_jsonl  # noqa: E402

AUTHORED_SOURCE = "llm_assisted"  # drafted with an LLM, gold labels assigned/reviewed by the maintainer


def row(id_, task, state, label, *, context=None, source=AUTHORED_SOURCE, variant_of=None, variant_type=None,
        note=None, **extra):
    r = {
        "id": id_, "task": task, "state": state, "context": context, "label": label, "label_b": None,
        "agreed": None, "source": source, "variant_of": variant_of, "variant_type": variant_type, "note": note,
    }
    r.update(extra)
    return r


# =============================================================================================== T1
# (label, text, note, t3_base). t3_base marks the 10 examples pre-selected for T3 *before* any model run,
# chosen because their gold label is comparatively unambiguous.
T1 = [
    ("ok", "本日発送しました。到着まで2〜3日ほどかかるかと思います。よろしくお願いします。", "通常連絡", True),
    ("ok", "さすがですね！梱包がとても丁寧で、箱を開けた瞬間うれしくなりました。またご縁があればお願いします。",
     "非皮肉の『さすが』", True),
    ("ok", "すみません、届いた品の角に少し傷がありました。写真を送るので、確認していただけますか。", "正当な苦情", True),
    ("ok", "お取り置きの件、もう少しだけ考えさせてください。明日の夜までにはお返事します。", "保留連絡", False),
    ("ok", "値下げは難しいとのこと、承知しました。今回は見送らせていただきますね。", "丁寧な辞退", False),
    ("ok", "この前いただいたマグカップ、毎朝使ってます。いい買い物でした〜", "お礼", False),
    ("ok", "発送が遅れてしまい申し訳ありません。体調を崩しておりました。本日必ず発送します。", "謝罪", False),
    ("ok", "受け取り評価まだでしたね、失礼しました！今つけておきます。", "通常連絡", False),
    ("harassment", "取引ありがとうございました〜！さすが、発送が早いですね😊（3週間待ちました）", "皮肉", True),
    ("harassment", "こんなに個性的な梱包、初めて見ました。新聞紙一枚で送ってくる方って本当にいるんですね。勉強になりました。",
     "褒め言葉形の皮肉", False),
    ("harassment", "お忙しいところ恐縮ですが、字が読めない方にもわかるように説明文を書き直していただけますか？",
     "敬語の侮辱", True),
    ("harassment", "この値段で出せる度胸、見習いたいです。ご自分でも買いたいと思われますか？", "皮肉", False),
    ("harassment", "評価、楽しみにしていてくださいね。住所はもう分かっていますので。", "婉曲な脅し", True),
    ("harassment", "お返事ありがとうございます。三日かけてその一文とは、よほどお忙しいんですね。", "皮肉", False),
    ("harassment", "素人さんは相場もご存じないんですね。まあ、そういう方がいるからこちらは助かりますけど。", "見下し", False),
    ("harassment", "次からは、日本語が通じる方とお取引しようと思います。", "婉曲な侮辱", False),
    ("scam", "この先のやり取りはアプリだと手数料がかかるので、こちらのメッセージアプリでお願いできますか？ID送りますね。",
     "外部誘導", True),
    ("scam", "在庫確保のため、先に代金の半分だけ銀行振込でお願いしています。確認でき次第お取り置きします。", "前払い要求", False),
    ("scam", "アカウント確認のため、ログインに使っているメールアドレスとパスワードを教えていただけますか？運営の者です。",
     "運営なりすまし", True),
    ("scam", "当選おめでとうございます！限定品の購入権が当たりました。本日中に下記リンクから手続きしてください。", "当選詐欺", False),
    ("scam", "直接取引なら2割引きにしますよ。振込先はあとで個別に送りますね。", "直接取引誘導", False),
    ("scam", "商品は発送済みです。まだ届いていないと思いますが、先に受取評価をしていただけると助かります。",
     "受取前評価の要求", False),
    ("scam", "お支払いがエラーになったようです。こちらのページでカード情報をもう一度入力してください。", "フィッシング", False),
    ("resale_spam", "同じ商品、あと30個あります。まとめ買いの方優先、業者さん歓迎です。", "大量出品", True),
    ("resale_spam", "発売日に並んで確保しました！定価の3倍ですが、今しか手に入りません。即決の方どうぞ。", "高額転売", True),
    ("resale_spam", "このシリーズを全部買い取ります。出品中のものがあれば何点でも。高く売れるので助かります。", "買い占め", False),
    ("resale_spam", "限定グッズを仕入れ値でお譲りします。転売していただいても大丈夫です、在庫はたくさんあるので。",
     "転売勧誘", False),
    ("resale_spam", "フォロワー限定で、抽選品を定価の倍でお分けしています。毎週入荷あり。", "高額転売", False),
    ("resale_spam", "出品されているぬいぐるみ、全部まとめて買うので値下げしてもらえますか？別のサイトで売る予定です。",
     "転売目的の買い占め", False),
    ("resale_spam", "在庫ありますよ〜。今週末限定、お一人様何個でもOK。次回入荷未定なのでお早めに。", "大量出品", False),
]

# =============================================================================================== T3
# Hand-written variants for each pre-selected T1 base, keyed by T1 index (1-based).
# katakana_halfwidth is derived from the hiragana variant mechanically.
T3 = {
    1: {
        "hiragana": "ほんじつはっそうしました。とうちゃくまで2〜3にちほどかかるかとおもいます。よろしくおねがいします。",
        "emoji_kaomoji": "本日発送しました📦✨ 到着まで2〜3日ほどかかるかと思います🙏 よろしくお願いします(^^)",
        "internet_slang": "今日発送したっす〜 着くまで2〜3日くらいかも。よろです",
    },
    2: {
        "hiragana": "さすがですね！こんぽうがとてもていねいで、はこをあけたしゅんかんうれしくなりました。またごえんがあればおねがいします。",
        "emoji_kaomoji": "さすがですね！😍 梱包がとても丁寧で、箱を開けた瞬間うれしくなりました(*´▽`*) またご縁があればお願いします🙇",
        "internet_slang": "さすがっす！梱包ガチ丁寧で、箱開けた瞬間テンション爆上がりでした。またよろです〜",
    },
    3: {
        "hiragana": "すみません、とどいたしなのかどにすこしきずがありました。しゃしんをおくるので、かくにんしていただけますか。",
        "emoji_kaomoji": "すみません💦 届いた品の角に少し傷がありました😢 写真を送るので、確認していただけますか🙏",
        "internet_slang": "すんません、届いたやつの角にちょい傷ありました。写真送るんで確認おなしゃす",
    },
    9: {
        "hiragana": "とりひきありがとうございました〜！さすが、はっそうがはやいですね😊（3しゅうかんまちました）",
        "emoji_kaomoji": "取引ありがとうございました〜！🙌 さすが、発送が早いですね😊😊（3週間待ちました）(^_^;)",
        "internet_slang": "取引あざした〜！さすが発送はっやｗ（3週間待ったけど）",
    },
    11: {
        "hiragana": "おいそがしいところきょうしゅくですが、じがよめないかたにもわかるようにせつめいぶんをかきなおしていただけますか？",
        "emoji_kaomoji": "お忙しいところ恐縮ですが🙇 字が読めない方にもわかるように説明文を書き直していただけますか？🤔📝",
        "internet_slang": "おつかれのとこアレですけど、字読めない人にもわかるように説明文書き直してもらえます？ｗ",
    },
    13: {
        "hiragana": "ひょうか、たのしみにしていてくださいね。じゅうしょはもうわかっていますので。",
        "emoji_kaomoji": "評価、楽しみにしていてくださいね😊 住所はもう分かっていますので🏠(^^)",
        "internet_slang": "評価、楽しみにしといてね〜。住所もう把握済みなんで",
    },
    17: {
        "hiragana": "このさきのやりとりはあぷりだとてすうりょうがかかるので、こちらのめっせーじあぷりでおねがいできますか？IDおくりますね。",
        "emoji_kaomoji": "この先のやり取りはアプリだと手数料がかかるので💸 こちらのメッセージアプリでお願いできますか？📱 ID送りますね✌",
        "internet_slang": "この先アプリだと手数料取られるんで、別のメッセアプリでやりとりしません？ID送るっす",
    },
    19: {
        "hiragana": "あかうんとかくにんのため、ろぐいんにつかっているめーるあどれすとぱすわーどをおしえていただけますか？うんえいのものです。",
        "emoji_kaomoji": "アカウント確認のため🔐 ログインに使っているメールアドレスとパスワードを教えていただけますか？🙏 運営の者です(｀・ω・´)ゞ",
        "internet_slang": "アカ確認のため、ログインのメアドとパス教えてもらえます？運営のもんです",
    },
    24: {
        "hiragana": "おなじしょうひん、あと30こあります。まとめがいのかたゆうせん、ぎょうしゃさんかんげいです。",
        "emoji_kaomoji": "同じ商品、あと30個あります📦📦📦 まとめ買いの方優先、業者さん歓迎です💰(^o^)",
        "internet_slang": "同じやつあと30個あるよ〜。まとめ買い優先、業者さん大歓迎っす",
    },
    25: {
        "hiragana": "はつばいびにならんでかくほしました！ていかの3ばいですが、いましかてにはいりません。そっけつのかたどうぞ。",
        "emoji_kaomoji": "発売日に並んで確保しました！🔥 定価の3倍ですが、今しか手に入りません⏰ 即決の方どうぞ👉✨",
        "internet_slang": "発売日に並んで確保したった！定価の3倍だけど今しか手に入らんよ。即決の人どうぞー",
    },
}


def hira_to_kata(s: str) -> str:
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in s)


def _halfwidth_table() -> dict[str, str]:
    table: dict[str, str] = {}
    for cp in range(0xFF61, 0xFFA0):
        h = chr(cp)
        full = unicodedata.normalize("NFKC", h)
        if len(full) == 1:
            table.setdefault(full, h)
        for mark in ("ﾞ", "ﾟ"):
            composed = unicodedata.normalize("NFKC", h + mark)
            if len(composed) == 1 and composed not in table:
                table[composed] = h + mark
    table["〜"] = "~"
    return table


HALFWIDTH = _halfwidth_table()


def to_halfwidth_katakana(s: str) -> str:
    return "".join(HALFWIDTH.get(c, c) for c in s)


# =============================================================================================== T2
# (level, text, note). 0 neutral / 1 dissatisfied but calm / 2 strongly angry (often under polite wording)
T2 = [
    (0, "商品受け取りました。ありがとうございました。", "お礼"),
    (0, "発送はいつ頃になりそうでしょうか。予定だけ教えていただけると助かります。", "中立の問い合わせ"),
    (0, "ご丁寧にありがとうございます。それでは明日の午前中に発送いたします。", "通常連絡"),
    (0, "写真を追加しましたので、ご確認ください。", "通常連絡"),
    (0, "お返事が遅くなってすみません。週末は家を空けておりました。", "謝罪"),
    (0, "値段交渉は可能ですか？難しければこのままでも大丈夫です。", "中立の問い合わせ"),
    (0, "無事に届いてよかったです。大切に使っていただけたらうれしいです。", "お礼"),
    (0, "在庫を確認しますので、少々お待ちください。", "通常連絡"),
    (0, "了解です、では駅前で18時にお願いします。", "通常連絡"),
    (0, "サイズ表記は実寸で測ったものです。参考になれば幸いです。", "通常連絡"),
    (1, "届いた商品ですが、説明にはなかった汚れが少しありました。次回からは気をつけていただけると助かります。", "穏やかな苦情"),
    (1, "発送予定日を過ぎているようですが、状況を教えていただけますか。", "穏やかな催促"),
    (1, "正直なところ、写真の印象とは少し違いました。まあ中古品なので仕方ないですね。", "落胆"),
    (1, "前回もお伝えしたのですが、時間指定は夜でお願いできればと思います。", "穏やかな念押し"),
    (1, "梱包が少し簡易的で、箱がつぶれていました。中身は無事だったので大丈夫です。", "穏やかな苦情"),
    (1, "お返事がなかなかいただけず、少し困っています。お手すきの際にご連絡ください。", "穏やかな催促"),
    (1, "残念ですが、今回は評価を『普通』にさせていただきます。", "落胆"),
    (1, "値下げ後すぐに別の方へ売られてしまったのは、少し残念でした。", "落胆"),
    (1, "付属品が一つ足りないようです。お手数ですがご確認いただけますか。", "穏やかな苦情"),
    (1, "説明文をもう少し詳しく書いていただけると、購入する側としては助かります。", "穏やかな要望"),
    (2, "何度ご連絡差し上げてもお返事がいただけないのは、どういうおつもりでしょうか。本日中にご回答がない場合、運営に報告いたします。",
     "丁寧語の最後通告"),
    (2, "大変恐縮ではございますが、これ以上の言い訳は結構です。代金を全額お返しください。", "丁寧語の強い抗議"),
    (2, "説明文に『美品』と書かれていたかと存じますが、こちらのどこが美品なのか、ぜひご教示いただけますでしょうか。",
     "丁寧語の皮肉"),
    (2, "ご丁寧なお詫び文、拝読いたしました。三度目ともなりますと、さすがに言葉もございません。", "丁寧語の憤り"),
    (2, "もう結構です。二度とお取引させていただくことはございませんので、ご安心ください。", "丁寧語の絶縁"),
    (2, "こちらは発送日を守ってお待ちしておりました。約束を守れないのであれば、最初から出品なさらないでいただけますか。",
     "丁寧語の強い抗議"),
    (2, "お言葉ですが、壊れた状態で送ってきたのはそちらです。責任を持って対応していただきます。", "強い抗議"),
    (2, "重ねて申し上げます。これ以上の遅延は一切容認いたしかねます。", "丁寧語の最後通告"),
    (2, "これで三回目の連絡です。いい加減にしていただけませんか。", "強い苛立ち"),
    (2, "誠意あるご対応を期待しておりましたが、期待した私が愚かでした。然るべき対応を取らせていただきます。", "丁寧語の憤り"),
]

# =============================================================================================== T4
# (label, context turns [(speaker, text)], target, note). `target` is said by the participant who did not
# speak last in `context`.
T4 = [
    ("accept", [("A", "18時に駅前で交換できますか？"), ("B", "18時半なら大丈夫です")], "それでお願いします", "主語・目的語省略"),
    ("accept", [("B", "こちらの本、おいくらをご希望ですか？"), ("A", "送料込みで2500円でいかがでしょう")], "それで大丈夫です",
     "省略"),
    ("accept", [("A", "土曜の午前中に取りに伺ってもいいですか？")], "ぜひ", "一語応答"),
    ("accept", [("B", "角に傷があるのが気になります"), ("A", "状態があまり良くないので、500円引きにしますね")], "助かります！",
     "間接的承諾"),
    ("accept", [("A", "箱なしでよければ少し安くできますが")], "なしで構いません", "省略"),
    ("accept", [("B", "青は売り切れてしまって…代わりにこちらの色違いでもよろしいですか？")], "むしろそっちが良かったです",
     "間接的承諾"),
    ("accept", [("B", "交換の品、そちらが不安でしたらこちらから先に送りましょうか？")], "じゃあお言葉に甘えて", "慣用表現"),
    ("accept", [("A", "3点とも気になっています"), ("B", "まとめて買っていただけるなら、3点で5000円にします")], "乗ります",
     "口語"),
    ("reject", [("A", "1000円まで下げてもらえませんか？")], "ちょっとそれは…", "言いさし"),
    ("reject", [("A", "今日中に発送してもらえますか？")], "今日はちょっと厳しいですね", "婉曲な断り"),
    ("reject", [("A", "手渡しでお願いできますか？")], "遠方なので、郵送のみとさせてください", "理由付きの断り"),
    ("reject", [("A", "来月買うので、それまで取り置きしてもらえますか？")], "すみません、お取り置きはしていないんです",
     "婉曲な断り"),
    ("reject", [("A", "動作確認の動画を撮っていただけませんか？")], "そこまではちょっと…写真でご判断いただければ",
     "婉曲な断り"),
    ("reject", [("A", "5000円で買います"), ("B", "6000円ではいかがでしょう"), ("A", "間を取って5500円でどうですか")],
     "うーん、6000円は譲れないです", "複数ターン"),
    ("reject", [("A", "交換、こちらのカードとではどうでしょう？")], "お気持ちだけいただいておきます", "慣用的な断り"),
    ("reject", [("A", "駅まで来ていただくことはできますか？")], "車がないもので、難しいかと…", "婉曲な断り"),
    ("question", [("A", "2000円でどうでしょう")], "送料込みですか？", "確認"),
    ("question", [("A", "明日の夕方に受け取りに行きますね")], "何時ごろになりそうですか？", "確認"),
    ("question", [("A", "交換はこちらのキーホルダーでいいですか")], "未開封ですか？", "確認"),
    ("question", [("B", "いくつか出品してますよね"), ("A", "セットでなら値下げします")], "どれとどれのセットですか？",
     "省略"),
    ("question", [("A", "来週の火曜なら発送できます")], "速達にもできますか？", "条件確認"),
    ("question", [("B", "届いた靴、少し小さかったです"), ("A", "代わりの品を送りますね")], "サイズは同じですか？", "省略"),
    ("question", [("A", "受け渡し場所、公園の入口でいいですか？")], "北口と南口、どちらですか？", "確認"),
    ("other", [("A", "3000円で譲っていただけますか？")], "少し考えさせてください", "保留"),
    ("other", [("A", "今週末に交換しませんか？")], "家族に予定を確認してから連絡します", "保留"),
    ("other", [("A", "値下げは可能ですか？")], "ちなみに、ほかにも似た商品を出品しています", "話題転換"),
    ("other", [("A", "明日発送でも大丈夫ですか？")], "こんばんは！お返事遅くなりました。", "あいさつのみ"),
    ("other", [("A", "こちらの本と交換しませんか？")], "そういえば、以前も一度お取引しましたよね", "話題転換"),
    ("other", [("A", "手渡しにしませんか？")], "ちょっと今バタバタしているので、夜にまた連絡します", "保留"),
    ("other", [("A", "セットで買ってくれるなら値引きしますよ")], "ありがとうございます、他の方の出品も見てから決めますね", "保留"),
]

# =============================================================================================== T5
# (label: refund requested?, clean text, note)
T5 = [
    (True, "先日購入したワイヤレスイヤホンですが、右側から音が出ません。出品者の方と連絡が取れないので、支払った代金を返してもらえないでしょうか。",
     "婉曲な返金要求"),
    (True, "届いた商品が注文したものと違いました。交換ではなく、キャンセルしてお金を戻していただきたいです。", "返金要求"),
    (True, "送料が二重に請求されているようです。多く引き落とされた分を戻していただけますか。", "一部返金"),
    (True, "発送から2週間経っても届きません。もう待てないので、取引を取り消して代金を返してください。", "返金要求"),
    (True, "説明では『未使用』とありましたが、明らかに使用感があります。一部でいいので返金に応じてもらえないか、出品者さんに伝えていただけますか。",
     "一部返金・仲介依頼"),
    (False, "届いたマグカップの取っ手が欠けていました。同じ商品があれば交換していただきたいです。お金はそのままで大丈夫です。",
     "交換要求"),
    (False, "発送通知が来ているのですが、追跡番号が表示されません。どこで確認できますか。", "質問"),
    (False, "以前の取引でお金を返してもらったことがあるのですが、その時の手続きは何日くらいかかりましたっけ？今回は問題なく届いたので念のための質問です。",
     "返金に言及するが要求なし"),
    (False, "受け取った商品に少し傷がありましたが、使う分には問題ないので、このままで大丈夫です。一応ご報告まで。", "報告"),
    (False, "出品者さんがとても親切で、到着も早かったです。お礼を伝える方法があれば教えてください。", "お礼"),
]

# Label-neutral distractors for T5 noise. Must not mention money, refunds, cancellation or returns.
NOISE_OPENERS = [
    "お世話になっております。いつもアプリを便利に使わせてもらっています。",
    "こんにちは。問い合わせフォームの使い方がよく分からず、長文になってしまったらすみません。",
    "夜分に失礼します。最近こちらは急に寒くなってきましたが、皆さまお元気でしょうか。",
    "はじめて問い合わせをします。普段はスマートフォンから見ているのですが、今日はパソコンから送っています。",
]
NOISE_ASIDES = [
    "ちなみに、先月プロフィール画像を変えようとしたときに、うまく反映されないことがありました。今は直っています。",
    "余談ですが、アプリのダークモードがとても見やすくて気に入っています。通知の音も変えられるといいなと思いました。",
    "話は変わりますが、出品画面で写真の順番を入れ替える方法がしばらく分からず、ヘルプを探してやっと見つけました。",
    "それと、検索でカテゴリを絞り込むと、たまに関係のない商品が出てくる気がします。気のせいかもしれません。",
    "家族もこのアプリを使い始めたのですが、アカウントを分けたほうがいいのか迷っているそうです。",
    "先週はお気に入り登録した商品の値下げ通知が届いて、とても便利でした。",
]
NOISE_CLOSERS = [
    "お手数をおかけしますが、よろしくお願いいたします。",
    "お忙しいところすみません。ご確認のほどよろしくお願いします。",
]
NOISE_SIGNATURES = [
    "――――――――――\n送信元: スマートフォンアプリ（バージョン 4.2.1）\n端末: Android 14",
    "――――――――――\n送信元: ブラウザ版\nOS: Windows 11 / ブラウザ: 最新版",
]
NOISE_QUOTED = [
    "> 【自動返信】お問い合わせを受け付けました。\n> 担当者より順番にご連絡いたします。\n> 内容によってはお時間をいただく場合がございます。\n> このメッセージは送信専用です。",
    "> 以前のお問い合わせ（プロフィール画像について）\n> 画像の形式を変えてもう一度お試しください。\n> 改善しない場合は、アプリを最新版に更新してください。",
    "> 以前のお問い合わせ（通知設定について）\n> 設定画面の「お知らせ」から個別に切り替えられます。\n> 端末側の通知許可もあわせてご確認ください。",
]


def t5_noisy(clean: str, level: str, rng: random.Random) -> str:
    if level == "medium":
        parts = [rng.choice(NOISE_OPENERS), clean, rng.choice(NOISE_ASIDES), rng.choice(NOISE_CLOSERS)]
        return "\n\n".join(parts)
    if level == "heavy":
        opener = rng.sample(NOISE_OPENERS, 2)
        asides = rng.sample(NOISE_ASIDES, 4)
        quoted = rng.sample(NOISE_QUOTED, 2)
        parts = [*opener, asides[0], asides[1], clean, asides[2], asides[3], rng.choice(NOISE_CLOSERS),
                 rng.choice(NOISE_SIGNATURES), *quoted]
        return "\n\n".join(parts)
    raise ValueError(level)


def build() -> dict[str, list[dict]]:
    t1 = [row(f"t1-{i:03d}", "t1_moderation", text, label, note=note, t3_base=base)
          for i, (label, text, note, base) in enumerate(T1, 1)]
    by_index = {i: r for i, r in enumerate(t1, 1)}

    t3 = []
    n = 0
    for idx, variants in T3.items():
        base = by_index[idx]
        assert base["t3_base"], idx
        texts = dict(variants)
        texts["katakana_halfwidth"] = to_halfwidth_katakana(hira_to_kata(variants["hiragana"]))
        for vtype in ["hiragana", "katakana_halfwidth", "emoji_kaomoji", "internet_slang"]:
            n += 1
            t3.append(row(f"t3-{n:03d}", "t3_surface_variants", texts[vtype], base["label"], source="derived",
                          variant_of=base["id"], variant_type=vtype, note=f"{base['id']} の {vtype} 変形"))

    t2 = [row(f"t2-{i:03d}", "t2_politeness_anger", text, level, note=note)
          for i, (level, text, note) in enumerate(T2, 1)]

    t4 = [row(f"t4-{i:03d}", "t4_ellipsis", target, label, note=note,
              context=[{"speaker": s, "text": t} for s, t in ctx])
          for i, (label, ctx, target, note) in enumerate(T4, 1)]

    t5 = []
    for i, (label, clean, note) in enumerate(T5, 1):
        base_id = f"t5-{i:03d}-clean"
        t5.append(row(base_id, "t5_noise", clean, label, variant_type="clean", note=note))
        rng = random.Random(SEED * 100 + i)
        for level in ("medium", "heavy"):
            t5.append(row(f"t5-{i:03d}-{level}", "t5_noise", t5_noisy(clean, level, rng), label, source="derived",
                          variant_of=base_id, variant_type=level, note=f"{base_id} に無関係な文脈を付加 ({level})"))

    return {"t1_moderation.jsonl": t1, "t2_politeness_anger.jsonl": t2, "t3_surface_variants.jsonl": t3,
            "t4_ellipsis.jsonl": t4, "t5_noise.jsonl": t5}


def main() -> None:
    for name, rows in build().items():
        write_jsonl(DATA_DIR / name, rows)
        print(f"{name}: {len(rows)}")


if __name__ == "__main__":
    main()
