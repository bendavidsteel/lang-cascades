"""Have a Claude model code the sampled stance targets, as a second annotator.

Takes the CSV written by ``sample_classified_posts.py`` and asks a model the same
two questions the coding page asks a person -- is the target relevant to the post,
and what is the post's stance on it -- then writes the answers in the coding page's
own export format, so ``score_coded_posts.py`` reads the result as another coder:

    python scripts/target_mining/score_coded_posts.py \
        -i out/stance_coding_coder_2026-07-29.csv -i out/stance_coding_llm_haiku.csv \
        --sample out/classified_post_sample.csv --output out/coding.tex

That run reports coder-vs-coder agreement between the person and the model, and
scores the classifier against the two coders' majority label.

The annotator is never shown the classifier's label or its class probabilities: it
has to be independent of the thing being evaluated for the agreement numbers to mean
anything. It is also text-only, while the person coding saw the post embedded and
could watch the video, so disagreement on transcript- and OCR-derived posts carries a
modality difference as well as a judgement one -- ``text_source`` separates them.

One request per post, carrying all of that post's sampled targets, which is what the
person saw too. The codebook is cached, so it is billed once per five minutes rather
than once per post.

Runs are resumable: pairs already in the output CSV are skipped, so an interrupted
run continues where it stopped.

Examples:
    python scripts/target_mining/annotate_posts_llm.py \
        --input out/classified_post_sample.csv --dry-run

    python scripts/target_mining/annotate_posts_llm.py \
        --input out/classified_post_sample.csv \
        --output out/stance_coding_llm_haiku.csv
"""

import argparse
import concurrent.futures
import csv
import datetime
import os
import re
import sys
import threading
from collections import OrderedDict

STANCES = ['FAVOR', 'AGAINST', 'NEUTRAL']

# the coding page's export columns, so a coded file from either source reads the same
EXPORT_COLUMNS = ['platform', 'post_id', 'post_url', 'seed_name', 'handle', 'main_type',
                  'sub_type', 'party', 'actor_group', 'year', 'createtime', 'target',
                  'model_stance', 'coded_relevant', 'coded_stance', 'note', 'coder',
                  'coded_at', 'n_post_targets', 'n_sampled_targets', 'pair_weight']

# never shown to the annotator: it is being scored against these
WITHHELD_COLUMNS = {'stance', 'model_stance', 'model_prob_neutral', 'model_prob_favor',
                    'model_prob_against'}

PRICING = {           # $ per million tokens
    'claude-haiku-4-5': (1.00, 0.10, 1.25, 5.00),
    'claude-sonnet-5': (2.00, 0.20, 2.50, 10.00),
    'claude-opus-5': (5.00, 0.50, 6.25, 25.00),
}

# what post_text actually is, per the sampler's text_source label
PROVENANCE = {
    'written': 'the text the author typed',
    'caption': 'the caption the author wrote on an image or video',
    'caption+transcript': "the author's caption, followed by an automatic "
                          'speech-to-text transcript of the video',
    'caption+ocr': "the author's caption, followed by text read off the image by OCR",
    'transcript': 'an automatic speech-to-text transcript of the video; the author '
                  'wrote no caption',
    'ocr': 'text read off the image by OCR; the author wrote no caption',
    'repost text': 'the text of a post this account reposted',
    'link card': 'the title and description of a link the post shares',
    'unknown': 'the captured post text; how it was captured was not recorded',
}


CODEBOOK = """\
You are coding social media posts for a stance-detection study. An automatic system \
read each post and pulled out phrases it believed the post takes a stance on. Your \
job is to check that work, one post at a time, exactly as a trained human coder would.

For every target listed with a post you answer two questions, independently.


# Question 1: is the target relevant to the post?

Relevant means the target is genuinely something this post is about -- a person, \
organisation, policy, event, group or idea the post actually discusses, addresses or \
takes a position on.

Mark a target relevant when:
- the post says something about it, however briefly, including simply reporting on it
- it is named in the post and the post's point concerns it
- it is not named outright but is unmistakably what the post is about (a post about \
"the carbon tax" is about carbon pricing even if it says "this tax")
- the post replies to or quotes another post about it, and engages with that subject

Mark a target NOT relevant when:
- the phrase was pulled out of the text but the post takes no position on it and it \
is not what the post is about -- a passing name-drop, a hashtag with no bearing on \
the content, a boilerplate sign-off, a URL fragment, the account's own handle
- the phrase is garbled, truncated or meaningless ("the thing", "https", "ottawa on")
- the phrase came from transcript or OCR noise rather than the post's content
- the phrase is so vague that "the post's stance on it" is not a coherent question
- it names something adjacent to the post's subject but not discussed in it

Two cases that come up constantly and are easy to get wrong:

- A repost, or a post that shares a news story, is about what it reposts. The account \
chose to circulate it. The subjects of the reposted material are relevant even when \
the account adds no words of its own -- relevance asks what the post is about, not \
how much the account wrote.
- A target may be worded from the conversation rather than lifted out of the post's \
own sentence, because targets are phrased by an automatic system and a reply is read \
together with what it replies to. If the post engages with that subject at all, it is \
relevant, even where the wording appears only in the parent post.

Extraction errors are the thing this study measures, so do not stretch to make a bad \
target work. If the target is not really what the post is about, say so. But do not \
mark a target irrelevant merely because the post is short, is a repost, or does not \
repeat the target's exact words.


# Question 2: what is the post's stance on the target?

This is the stance of the account that posted, toward the target. Not your own view, \
not whether the claim is true, not whether the post is positive or negative in tone \
overall.

FAVOR   -- the post supports, endorses, defends, praises or advocates for the target,
           or argues for a position the target stands for.
AGAINST -- the post opposes, criticises, attacks, mocks, blames or argues against the
           target, or argues for a position the target opposes.
NEUTRAL -- the post mentions or discusses the target without taking a side: neutral
           reporting, a factual statement, a question, an announcement, an even-handed
           description, or a case where a stance is genuinely unclear from the text.

Answer question 2 for every target, including targets you marked not relevant. The \
two answers are recorded separately and one is never inferred from the other. Where a \
target is not relevant, give the stance the text would most support if the target were \
taken at face value, and fall back to NEUTRAL when nothing supports a side.

Judgement calls that come up often:
- Sarcasm and irony are common and usually signal AGAINST. "Great job as always, \
minister" after a list of failures is AGAINST, not FAVOR.
- Quoting or amplifying someone else's words does not make them the account's view. \
A post quoting an opponent to condemn them is AGAINST that opponent. A repost with no \
comment usually carries the quoted post's stance.
- Attacking a rival is not the same as supporting their opponent. Code the stance on \
the target in front of you, and only mark FAVOR for a third party if the post actually \
endorses them.
- Criticising a government's handling of an issue is AGAINST the government. It says \
nothing on its own about the issue, which may well be NEUTRAL or FAVOR.
- Calling for something to change, be funded, be scrapped or be investigated is a \
stance: FAVOR the change, and usually AGAINST the thing being scrapped.
- A post can be strongly worded and still NEUTRAL toward a particular target if that \
target is only the setting for the argument.
- Emoji, and interjections such as "wow" or "unreal", carry stance. Read them.


A warning about NEUTRAL. The most common mistake on this task is reading warm or \
confident tone as support. Politicians write in a register that sounds positive by \
default: announcements, process updates, thanks, turns of phrase like "robust", \
"meaningful", "proud to". None of that is a stance on its own. Ask what position the \
post takes that someone could disagree with. If the answer is "none -- it is telling \
people that something happened", the stance is NEUTRAL however upbeat the wording.

Reserve FAVOR for posts that argue for the target, defend it, promote it as good, or \
claim it as an achievement worth backing. Reserve AGAINST for posts that argue \
against it, blame it, or hold it up as a failure. Everything else is NEUTRAL, which \
is a common and correct answer, not a last resort. Do not spread FAVOR and AGAINST \
across a post's targets to avoid repeating yourself: two targets in one post often \
take the same label, and sometimes all of them do.


# The shapes targets come in

Targets are short noun phrases pulled out automatically, so they vary in how usable \
they are:

- A named person, party or organisation ("pierre poilievre", "npd", "elections \
canada"). Usually easy: relevant if the post is about them, and the stance is \
whatever the post says about them.
- A policy or programme ("carbon tax", "dental care plan", "bill 21"). Relevant if the \
post discusses the policy, not merely something in the same area. A post about a \
hospital wait time is not about "healthcare funding" unless it raises the funding.
- An abstraction or value ("democracy", "affordability", "government openness"). \
These need care. They are relevant only where the post actually engages with the \
idea, and the stance is toward the idea, not toward whoever invoked it. A post \
attacking a minister who claims to defend affordability is AGAINST the minister and \
usually NEUTRAL on affordability.
- A place or institution ("saskatchewan", "riverside public school"). Often the \
setting rather than the subject: relevant, but NEUTRAL, unless the post praises or \
criticises the place itself.
- An event ("swearing-in ceremony", "the convoy"). Stance is how the post treats the \
event -- celebrating it is FAVOR, condemning it is AGAINST, announcing it is NEUTRAL.
- A fragment or noise ("ottawa on", "the thing", "https", "point action"). Not \
relevant. These are what the study is looking for.

Hashtags are part of the post and can carry stance (#StopTheCarbonTax is AGAINST), but \
a wall of unrelated promotional hashtags is not a subject the post is about.


# The material you are given

Posts come from Canadian political accounts -- elected politicians, parties, \
candidates, commentators and influencers -- on Twitter/X, Bluesky, Instagram and \
TikTok, from 2022 onward. Expect federal and provincial party politics (Liberal, \
Conservative, NDP, Bloc Quebecois, Green, People's Party), party leaders and premiers, \
carbon pricing, housing, healthcare, immigration, energy and pipelines, encampments, \
the convoy, Quebec sovereignty and language law, and Canadian commentary on US events. \
Some posts are in French, some mix both languages; code them the same way. \
Targets are always given in English, so a French post's targets have been translated: \
match them to the French wording they stand for rather than expecting them verbatim.

The post text is not always typed by the author, and each post says where its text \
came from:
- a transcript is machine speech-to-text: it has no punctuation you can trust, it \
mishears names, and it may run past the point where the post's argument ends
- OCR text is read off an image: it is fragmentary, it picks up captions, watermarks \
and interface chrome, and its line order may be wrong
- a caption followed by a transcript or OCR means the author's own words come first \
and the machine-read text follows

Treat machine-read text as evidence of what the post says, not as the author's \
phrasing. Targets that exist only because of transcription noise are not relevant.

When a parent post is shown, the post you are coding is a reply to it or quotes it. \
The parent is context for reading the post -- code the stance of the post itself, not \
the parent's.

You may be shown the account's party and role. Use it to resolve who "we" and "they" \
refer to, but do not let it decide the stance: politicians criticise their own side \
and praise their opponents often enough to matter.


# Worked examples

These are illustrations, not posts from the sample.

Post (Conservative MP, Twitter): "Another 100,000 Canadians used a food bank last \
month. Eight years of this government's failed policies. Canadians deserve better."
  target "food bank"  -> relevant, NEUTRAL. The post reports food bank use to make a
     point about the government; it takes no position on food banks themselves.
  target "this government" -> relevant, AGAINST. Explicit blame.

Post (NDP account, TikTok, caption + transcript): "we've been saying this for years \
[transcript] ...so when we talk about pharmacare what we're really talking about is \
whether a kid with diabetes has to ration insulin and I think most people in this \
country already know the answer to that"
  target "pharmacare" -> relevant, FAVOR. The argument is made in favour of it.
  target "this country" -> not relevant, NEUTRAL. Transcript filler, not a subject
     the post takes a position on.

Post (commentator, Bluesky, quoting a premier's announcement): "Ah yes, nothing says \
'fiscal responsibility' like a billion dollar giveaway to your donors. Brilliant stuff."
  target "fiscal responsibility" -> relevant, NEUTRAL. The phrase is quoted
     sarcastically; the post's target is the spending, not the principle.
  target "billion dollar giveaway" -> relevant, AGAINST. Sarcasm reading as criticism.

Post (Liberal MP, Instagram caption): "Great to join students at Riverside Public \
School this morning to talk about our national school food program."
  target "national school food program" -> relevant, FAVOR. Promoted as the MP's own.
  target "riverside public school" -> relevant, NEUTRAL. Mentioned as the setting.

Post (commentator, Bluesky, repost text with no added comment): "Reuters: the premier \
has asked Ottawa to extend the funding deal by two years, citing delays in hiring."
  target "the premier" -> relevant, NEUTRAL. A reposted news item the account chose to
     circulate is about its subjects, and this one reports rather than argues. Short
     and wordless is not a reason to call it irrelevant.
  target "funding deal" -> relevant, NEUTRAL. Reported, with no side taken.

Post (Bloc MP, Bluesky, replying to a post about federal health transfers): "Encore \
une fois, Ottawa décide seul et envoie la facture aux provinces. Québec mérite mieux."
  target "federal health transfers" -> relevant, AGAINST. The targets are translated
     into English, so match them to the French wording: the post attacks how Ottawa
     handles the transfers the parent post raised.
  target "quebec" -> relevant, FAVOR. "Québec mérite mieux" defends Quebec's position.

Post (Conservative candidate, Instagram, caption + OCR): "Doors knocked today! \
[OCR] VOTE RECORD TURNOUT ELECTIONS CANADA ... NG TURNO RECOR ... follow for more"
  target "elections canada" -> not relevant, NEUTRAL. The phrase is interface text and
     a broken OCR fragment, not something the post discusses.
  target "door knocking" -> relevant, FAVOR. The caption is the author's own words and
     presents the campaigning positively.

Post (Liberal backbencher, Twitter): "I've said this to my own caucus: the housing \
file has been mismanaged for a decade, and pretending otherwise insults every young \
person priced out of a home."
  target "housing file" -> relevant, AGAINST. The post condemns how it has been run,
     and the account's own party affiliation does not soften that.
  target "young person" -> relevant, FAVOR. Invoked sympathetically, as the people
     being failed.

Post (premier, Twitter): "Our government will not apologise for standing up for \
workers. The union leadership can keep playing politics."
  target "workers" -> relevant, FAVOR. Claimed as the thing being defended.
  target "union leadership" -> relevant, AGAINST. Accused of playing politics.
     Defending workers does not make the post FAVOR their union: code each target on
     what the post says about that target.


# When the post is thin

Plenty of posts in this corpus are a single line, a photo caption, or a transcript \
that wanders. Code what is in front of you rather than what the post probably meant:

- Short is not the same as empty. "Proud to support this." with a target naming what \
"this" is, is FAVOR.
- Where the text gives no way to tell, NEUTRAL is the answer, and relevance still \
depends on whether the target is what the post is about.
- Do not import outside knowledge about the account's party to supply a stance the \
text does not carry. Knowing a Conservative MP probably opposes carbon pricing is not \
evidence that this post does.
- Do not infer a stance from the target's own wording. A target phrased as "trudeau's \
failed policies" is not AGAINST unless the post argues against them; the phrasing \
came from an automatic system, not from the author.


# Recording your answers

Call the record_annotations tool once, with one entry per target listed, using each \
target's text exactly as it was given to you, in the order given. Write the short \
reason first and then the two answers. Keep the reason to one or two sentences \
pointing at what in the post decided it.

Code every target you are given, even where the post text is thin, empty or \
unreadable. If there is genuinely nothing to go on, mark the target not relevant with \
stance NEUTRAL and say so in the reason."""


ANNOTATION_TOOL = {
    'name': 'record_annotations',
    'description': 'Record the relevance and stance judgement for every target listed '
                   'with the post. One entry per target, in the order given.',
    'input_schema': {
        'type': 'object',
        'properties': {
            'annotations': {
                'type': 'array',
                'description': 'One entry per target listed with the post.',
                'items': {
                    'type': 'object',
                    'properties': {
                        'target': {
                            'type': 'string',
                            'description': 'The target, exactly as it was listed.'},
                        'reason': {
                            'type': 'string',
                            'description': 'One or two sentences on what in the post '
                                           'decided the two answers.'},
                        'relevant': {
                            'type': 'boolean',
                            'description': 'Whether the target is genuinely something '
                                           'this post is about.'},
                        'stance': {
                            'type': 'string',
                            'enum': STANCES,
                            'description': "The post's stance on the target, answered "
                                           'whether or not it is relevant.'},
                    },
                    'required': ['target', 'reason', 'relevant', 'stance'],
                    'additionalProperties': False,
                },
            },
        },
        'required': ['annotations'],
        'additionalProperties': False,
    },
}


def _clean(value):
    return re.sub(r'\s+\n', '\n', (value or '').strip())


def format_post(post):
    """The user message for one post: who posted it, the text, and the targets."""
    role = ' / '.join(v for v in [post.get('main_type'), post.get('sub_type')] if v)
    region = ', '.join(v for v in [post.get('electoral_district'),
                                   post.get('province')] if v)
    account = [
        f"name: {post.get('seed_name') or 'unknown'}"
        + (f" (@{post['handle']})" if post.get('handle') else ''),
        f"platform: {post.get('platform')}",
    ]
    if role:
        account.append(f'role: {role}')
    if post.get('party'):
        account.append(f"party: {post['party']}")
    if region:
        account.append(f'region: {region}')
    if post.get('createtime'):
        account.append(f"posted: {post['createtime'][:10]}")

    parts = ['<account>', '\n'.join(account), '</account>']

    parent = _clean(post.get('parent_text'))
    if parent:
        parts += ['', '<parent_post>',
                  'The post below is a reply to, or quotes, this post.', '',
                  parent, '</parent_post>']

    source = (post.get('text_source') or 'unknown').strip() or 'unknown'
    text = _clean(post.get('post_text'))
    parts += ['', f'<post_text source="{source}">',
              f'This is {PROVENANCE.get(source, PROVENANCE["unknown"])}.', '',
              text or '(no text was captured for this post)', '</post_text>']

    targets = '\n'.join(f'{i}. {t}' for i, t in enumerate(post['targets'], 1))
    parts += ['', '<targets>', targets, '</targets>', '',
              f"Code {'each' if len(post['targets']) > 1 else 'the'} target above "
              'for relevance and stance, and record your answers with the '
              'record_annotations tool.']
    return '\n'.join(parts)


def read_sample(path):
    """Group the sampler's long-format rows into posts, targets in file order."""
    with open(path, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f'{path} is empty')
    missing = {'platform', 'post_id', 'target', 'post_text'} - set(rows[0])
    if missing:
        raise SystemExit(f'{path} is missing columns: {sorted(missing)}')

    posts = OrderedDict()
    for row in rows:
        key = (row['platform'], row['post_id'])
        post = posts.get(key)
        if post is None:
            post = {k: v for k, v in row.items() if k not in WITHHELD_COLUMNS}
            post['targets'] = []
            post['rows'] = {}
            posts[key] = post
        if row['target'] not in post['rows']:
            post['targets'].append(row['target'])
            post['rows'][row['target']] = row
    return list(posts.values())


def parse_annotations(payload, targets):
    """Line the model's entries up with the targets asked about, or say what is wrong."""
    entries = payload.get('annotations')
    if not isinstance(entries, list):
        return None, 'no annotations array'

    by_target = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = (entry.get('target') or '').strip()
        stance = (entry.get('stance') or '').strip().upper()
        if stance not in STANCES or not isinstance(entry.get('relevant'), bool):
            continue
        by_target[name.lower()] = (entry['relevant'], stance,
                                   _clean(entry.get('reason'))[:500])

    out, missing = {}, []
    for target in targets:
        found = by_target.get(target.strip().lower())
        if found is None:
            missing.append(target)
        else:
            out[target] = found
    if missing:
        return None, f'no usable answer for {missing}'
    return out, None


def annotate_post(client, post, model, max_tokens, attempts=3):
    """One request per post. Returns (answers, usage); raises on a run of failures."""
    message = format_post(post)
    last = None
    for attempt in range(attempts):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            # the SDK dropped the typed temperature argument; the API still takes it
            extra_body={'temperature': 0},
            system=[{'type': 'text', 'text': CODEBOOK,
                     'cache_control': {'type': 'ephemeral'}}],
            tools=[ANNOTATION_TOOL],
            tool_choice={'type': 'tool', 'name': ANNOTATION_TOOL['name']},
            messages=[{'role': 'user', 'content': message}],
        )
        usage = response.usage
        for block in response.content:
            if block.type == 'tool_use':
                answers, problem = parse_annotations(block.input, post['targets'])
                if answers:
                    return answers, usage
                last = problem
                break
        else:
            last = f'no tool call (stop_reason {response.stop_reason})'
        if attempt + 1 < attempts:
            message = (format_post(post) + '\n\nYour previous answer could not be '
                       f'read: {last}. Answer again, with one entry per target, '
                       'using each target exactly as listed.')
    raise RuntimeError(f"{post['platform']}/{post['post_id']}: {last}")


def output_row(post, target, answer, coder, coded_at):
    relevant, stance, reason = answer
    row = {c: (post.get(c) or '') for c in EXPORT_COLUMNS}
    source = post['rows'][target]
    row.update({
        'target': target,
        'model_stance': source.get('stance', ''),
        'coded_relevant': '1' if relevant else '0',
        'coded_stance': stance,
        'note': reason,
        'coder': coder,
        'coded_at': coded_at,
    })
    for column in ['n_post_targets', 'n_sampled_targets', 'pair_weight']:
        row[column] = source.get(column, '')
    return row


def done_pairs(path):
    """Pairs already coded in an output file, so an interrupted run can continue."""
    if not os.path.exists(path):
        return set()
    with open(path, newline='', encoding='utf-8') as handle:
        return {(r['platform'], r['post_id'], r['target'])
                for r in csv.DictReader(handle)
                if (r.get('coded_stance') or '').strip()}


def report_cost(totals, model):
    rates = PRICING.get(model)
    line = (f"tokens: {totals['input']:,} input, {totals['cache_write']:,} cache write, "
            f"{totals['cache_read']:,} cache read, {totals['output']:,} output")
    if not rates:
        return line
    fresh, read, write, out = rates
    cost = (totals['input'] * fresh + totals['cache_read'] * read
            + totals['cache_write'] * write + totals['output'] * out) / 1e6
    return f'{line}\ncost at {model} list prices: ${cost:.2f}'


def dry_run(posts, show):
    """Print what would be sent, and what it would cost, without calling the API."""
    chars = sum(len(format_post(p)) for p in posts)
    pairs = sum(len(p['targets']) for p in posts)
    for post in posts[:show]:
        print('=' * 78)
        print(format_post(post))
    print('=' * 78)
    print(f'{len(posts)} posts, {pairs} pairs')
    print(f'codebook ~{len(CODEBOOK) // 4:,} tokens (cached), '
          f'post messages ~{chars // 4:,} tokens in total')
    print('the codebook has to clear the model\'s minimum cacheable prefix to be '
          'cached at all -- check cache_read on the first real run')


def selftest():
    post = {'platform': 'tiktok', 'post_id': '1', 'seed_name': 'A Politician',
            'handle': 'apol', 'main_type': 'politician', 'sub_type': 'mp',
            'party': 'NDP', 'province': 'Ontario', 'electoral_district': '',
            'createtime': '2024-03-02T10:00:00.000000+0000',
            'text_source': 'caption+transcript', 'post_text': 'a caption\nwhat was said',
            'parent_text': '', 'targets': ['pharmacare', 'this country'],
            'rows': {'pharmacare': {'stance': 'FAVOR', 'n_post_targets': '3'},
                     'this country': {'stance': 'NEUTRAL', 'n_post_targets': '3'}}}

    message = format_post(post)
    assert 'speech-to-text transcript' in message
    assert '1. pharmacare\n2. this country' in message
    assert 'parent_post' not in message
    assert 'FAVOR' not in message, 'the classifier label must not reach the annotator'

    post['parent_text'] = 'the post being replied to'
    assert 'the post being replied to' in format_post(post)

    answers, problem = parse_annotations(
        {'annotations': [{'target': 'This Country', 'relevant': False,
                          'stance': 'neutral', 'reason': 'filler'},
                         {'target': 'pharmacare', 'relevant': True,
                          'stance': 'FAVOR', 'reason': 'argues for it'}]},
        post['targets'])
    assert problem is None and answers['this country'] == (False, 'NEUTRAL', 'filler')
    assert answers['pharmacare'][1] == 'FAVOR'

    _, problem = parse_annotations({'annotations': [
        {'target': 'pharmacare', 'relevant': True, 'stance': 'MAYBE', 'reason': ''}]},
        post['targets'])
    assert problem and 'pharmacare' in problem

    row = output_row(post, 'pharmacare', (True, 'FAVOR', 'argues for it'),
                     'haiku-4.5', '2026-09-14T00:00:00Z')
    assert set(row) == set(EXPORT_COLUMNS)
    assert row['model_stance'] == 'FAVOR' and row['coded_relevant'] == '1'
    print('selftest: prompt, parsing and output rows pass')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', default='out/classified_post_sample.csv',
                        help='long-format CSV written by sample_classified_posts.py')
    parser.add_argument('-o', '--output',
                        help='where to write the coded CSV (default: named after the '
                             'model, beside the input)')
    parser.add_argument('-m', '--model', default='claude-haiku-4-5',
                        help='model to annotate with')
    parser.add_argument('--coder',
                        help='coder name recorded in the CSV (default: the model)')
    parser.add_argument('-c', '--concurrency', type=int, default=8,
                        help='posts in flight at once')
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--limit', type=int,
                        help='only annotate the first N posts, for a trial run')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the prompts and the token estimate, call nothing')
    parser.add_argument('--show', type=int, default=2,
                        help='posts to print in full with --dry-run')
    parser.add_argument('--selftest', action='store_true',
                        help='check the prompt and parsing rules and exit')
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    posts = read_sample(args.input)
    if args.limit:
        posts = posts[:args.limit]

    if args.dry_run:
        dry_run(posts, args.show)
        return

    output = args.output or os.path.join(
        os.path.dirname(args.input),
        f"stance_coding_llm_{args.model.replace('claude-', '')}.csv")
    coder = args.coder or args.model.replace('claude-', '')

    done = done_pairs(output)
    todo = [p for p in posts if any((p['platform'], p['post_id'], t) not in done
                                    for t in p['targets'])]
    if done:
        print(f'{len(done)} pair(s) already coded in {output}, {len(todo)} post(s) left')
    if not todo:
        print('nothing to do')
        return

    try:
        import anthropic
    except ImportError:
        raise SystemExit('pip install anthropic (or: uv run --with anthropic ...)')
    client = anthropic.Anthropic()

    totals = dict(input=0, cache_write=0, cache_read=0, output=0)
    lock = threading.Lock()
    fresh = not os.path.exists(output) or os.path.getsize(output) == 0
    handle = open(output, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(handle, fieldnames=EXPORT_COLUMNS)
    if fresh:
        writer.writeheader()

    def run(post):
        answers, usage = annotate_post(client, post, args.model, args.max_tokens)
        coded_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with lock:
            for target in post['targets']:
                if (post['platform'], post['post_id'], target) in done:
                    continue
                writer.writerow(output_row(post, target, answers[target], coder, coded_at))
            handle.flush()
            totals['input'] += usage.input_tokens
            totals['cache_write'] += usage.cache_creation_input_tokens or 0
            totals['cache_read'] += usage.cache_read_input_tokens or 0
            totals['output'] += usage.output_tokens
        return post

    def progress(i):
        if sys.stderr.isatty():
            print(f'{i}/{len(todo)} posts', end='\r', file=sys.stderr, flush=True)

    failures = []
    try:
        # one post first, so the rest of the run reads the codebook from cache
        run(todo[0])
        progress(1)
        with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
            futures = {pool.submit(run, p): p for p in todo[1:]}
            for i, future in enumerate(concurrent.futures.as_completed(futures), 2):
                try:
                    future.result()
                except Exception as error:                # keep going, report at the end
                    failures.append(str(error))
                progress(i)
    finally:
        handle.close()

    print(f'wrote {output}')
    print(report_cost(totals, args.model))
    if totals['cache_read'] == 0:
        print('note: the codebook is under this model\'s minimum cacheable prefix, so '
              'every request paid for it in full', file=sys.stderr)
    if failures:
        print(f'{len(failures)} post(s) failed; rerun to retry them:', file=sys.stderr)
        for failure in failures[:10]:
            print(f'  {failure}', file=sys.stderr)


if __name__ == '__main__':
    main()
