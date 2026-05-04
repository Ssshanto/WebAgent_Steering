"""Action-interface prompt variants for grounding experiments.

The variants rewrite only the interface identifiers shown to the model.  They
keep a reversible map back to BrowserGym's real bids so stepped MiniWob runs can
execute normally while frozen runs can score the shown action interface.
"""

from dataclasses import dataclass, field
import random
import re
import uuid

AX_START = "# Current page Accessibility Tree\n"
DOM_START = "\n\n# Current page DOM\n"
ACTION_START = "\n\n# Action Space\n"

ACTION_RE = re.compile(r"^\s*([A-Za-z_]\w*)\((.*)\)\s*$", re.S)
CALL_RE = re.compile(r"([A-Za-z_]\w*)\(([^)]*)\)")
QUOTED_RE = re.compile(r"""['"]([^'"]+)['"]""")

INTERFACE_MODES = (
    "original",
    "permuted",
    "alphanumeric",
    "structured",
    "uuid",
    "handle",
    "mixed",
    "longprefix",
    "stale_ids",
    "fake_examples",
    "decoy_labels",
)

INTERFACE_MODE_ALIASES = {
    "orig": "original",
    "alnum": "alphanumeric",
    "stale": "stale_ids",
    "fake": "fake_examples",
    "decoy": "decoy_labels",
}

KNOWN_ACTIONS = {
    "click",
    "dblclick",
    "hover",
    "fill",
    "keyboard_press",
    "press",
    "select_option",
    "check",
    "uncheck",
    "focus",
    "clear",
    "drag_and_drop",
    "upload_file",
}

ID_ARGUMENT_ACTIONS = {
    "click",
    "dblclick",
    "hover",
    "fill",
    "select_option",
    "check",
    "uncheck",
    "focus",
    "clear",
    "drag_and_drop",
    "upload_file",
    "press",
}


@dataclass
class InterfaceTransform:
    mode: str
    real_to_shown: dict[str, str]
    shown_to_real: dict[str, str]
    current_ids: list[str]
    labels: dict[str, str] = field(default_factory=dict)
    stale_ids: set[str] = field(default_factory=set)
    fake_example_ids: set[str] = field(default_factory=set)
    action_example_ids: set[str] = field(default_factory=set)
    decoy_label_ids: set[str] = field(default_factory=set)

    def labels_for_metrics(self):
        return dict(self.labels)


def normalize_interface_mode(mode):
    mode = (mode or "original").strip().lower()
    mode = INTERFACE_MODE_ALIASES.get(mode, mode)
    if mode not in INTERFACE_MODES:
        raise ValueError(f"unknown interface mode: {mode}")
    return mode


def parse_interface_modes(text, default=("original",)):
    if not text:
        return list(default)
    modes = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            modes.append(normalize_interface_mode(part))
    return modes or list(default)


def interface_cache_tag(modes):
    abbrev = {
        "original": "original",
        "permuted": "permuted",
        "alphanumeric": "alnum",
        "structured": "structured",
        "uuid": "uuid",
        "handle": "handle",
        "mixed": "mixed",
        "longprefix": "longprefix",
        "stale_ids": "stale",
        "fake_examples": "fake_examples",
        "decoy_labels": "decoy_labels",
    }
    return "interface_" + "_".join(abbrev[normalize_interface_mode(m)] for m in modes)


def prompt_sections(prompt):
    ax0 = prompt.find(AX_START)
    dom0 = prompt.find(DOM_START)
    act0 = prompt.find(ACTION_START)
    if ax0 < 0 or dom0 < 0 or act0 < 0:
        return "", "", ""
    ax = prompt[ax0 + len(AX_START):dom0]
    dom = prompt[dom0 + len(DOM_START):act0]
    action = prompt[act0:]
    return ax, dom, action


def ids_from_axtree_text(axtree_text):
    return re.findall(r"\[([^\]\s]+)\]", axtree_text or "")


def labels_by_id(axtree_text):
    labels = {}
    for line in (axtree_text or "").splitlines():
        match = re.search(r"\[([^\]\s]+)\](.*)", line)
        if match:
            labels[match.group(1)] = match.group(2).strip()
    return labels


def parse_action(action):
    match = ACTION_RE.match(str(action or "").strip())
    if not match:
        return None
    quoted = QUOTED_RE.findall(match.group(2))
    return {
        "type": match.group(1),
        "bid": quoted[0] if quoted else "",
        "quoted_args": quoted,
    }


def _dedupe_ids(ids):
    return list(dict.fromkeys(str(x) for x in ids))


def _uuid_id(seed, idx, real):
    return "u_" + uuid.uuid5(uuid.NAMESPACE_URL, f"webagent:{seed}:{idx}:{real}").hex


def _shown_id_for(mode, seed, idx, real):
    if mode == "alphanumeric":
        return f"x{idx:02d}"
    if mode == "structured":
        return f"node-{idx:03d}"
    if mode == "uuid":
        return _uuid_id(seed, idx, real)
    if mode == "handle":
        return f"@h{idx:03d}"
    if mode == "longprefix":
        return f"current-interface-element-{idx:03d}-handle"
    if mode == "mixed":
        choices = [
            f"x{idx:02d}",
            f"node-{idx:03d}",
            f"@h{idx:03d}",
            _uuid_id(seed, idx, real),
            f"current-interface-element-{idx:03d}-handle",
        ]
        return choices[idx % len(choices)]
    return str(real)


def remap_ids(ids, mode, seed):
    mode = normalize_interface_mode(mode)
    ids = _dedupe_ids(ids)
    effective_mode = mode
    if mode in {"stale_ids", "fake_examples", "decoy_labels"}:
        effective_mode = "original"

    if effective_mode == "original":
        presented = ids[:]
    elif effective_mode == "permuted":
        presented = ids[:]
        rng = random.Random(seed)
        rng.shuffle(presented)
        if len(ids) > 1 and presented == ids:
            presented = presented[1:] + presented[:1]
    else:
        presented = [_shown_id_for(effective_mode, seed, idx, real) for idx, real in enumerate(ids)]

    real_to_shown = dict(zip(ids, presented))
    shown_to_real = {shown: real for real, shown in real_to_shown.items()}
    return real_to_shown, shown_to_real


def _replace_current_ids(ax, dom, real_to_shown):
    def replace_bracket(match):
        bid = match.group(1)
        return f"[{real_to_shown.get(bid, bid)}]"

    def replace_attr(match):
        attr, quote, bid = match.groups()
        return f"{attr}={quote}{real_to_shown.get(bid, bid)}{quote}"

    ax = re.sub(r"\[([^\]\s]+)\]", replace_bracket, ax)
    dom = re.sub(r"\b(bid|data-bid)=(['\"])([^'\"]+)\2", replace_attr, dom)
    return ax, dom


def _extract_action_example_ids(action_section):
    out = set()
    for _name, args in CALL_RE.findall(action_section or ""):
        quoted = QUOTED_RE.findall(args)
        if quoted:
            out.add(quoted[0])
    return out


def _with_distractors(ax, action_section, mode, seed, labels):
    stale_ids = set()
    fake_example_ids = set()
    decoy_label_ids = set()

    if mode == "stale_ids":
        stale_ids = {f"stale-{seed % 997:03d}-{i:02d}" for i in range(3)}
        stale_text = ", ".join(sorted(stale_ids))
        ax = (
            ax.rstrip()
            + "\n\nPrevious interface ids are stale and non-executable: "
            + stale_text
            + "\n"
        )

    if mode == "fake_examples":
        fake_example_ids = {f"fake-{seed % 997:03d}-{i:02d}" for i in range(3)}
        fake = sorted(fake_example_ids)
        action_section = (
            action_section.rstrip()
            + "\n\nInvalid examples from another page, do not copy: "
            + f"click('{fake[0]}'), fill('{fake[1]}', 'example'), select_option('{fake[2]}', 'A')\n"
        )

    if mode == "decoy_labels":
        decoy_label_ids = {f"label-id-{seed % 997:03d}-{i:02d}" for i in range(max(1, len(labels)))}
        replacements = {}
        for idx, bid in enumerate(labels):
            decoy = f"label-id-{seed % 997:03d}-{idx:02d}"
            replacements[bid] = decoy
            labels[bid] = f"{labels[bid]} visible-label-token {decoy}"
        lines = []
        for line in ax.splitlines():
            match = re.search(r"\[([^\]\s]+)\]", line)
            if match and match.group(1) in replacements:
                line = f"{line} visible-label-token {replacements[match.group(1)]}"
            lines.append(line)
        ax = "\n".join(lines)

    return ax, action_section, stale_ids, fake_example_ids, decoy_label_ids


def apply_interface_variant(prompt, mode="original", seed=0):
    mode = normalize_interface_mode(mode)
    ax0 = prompt.find(AX_START)
    dom0 = prompt.find(DOM_START)
    act0 = prompt.find(ACTION_START)
    if ax0 < 0 or dom0 < 0 or act0 < 0:
        transform = InterfaceTransform(mode, {}, {}, [], {})
        return prompt, transform

    before_ax = prompt[: ax0 + len(AX_START)]
    ax = prompt[ax0 + len(AX_START):dom0]
    between = prompt[dom0: dom0 + len(DOM_START)]
    dom = prompt[dom0 + len(DOM_START):act0]
    before_action = prompt[act0: act0 + len(ACTION_START)]
    action_section = prompt[act0 + len(ACTION_START):]

    real_ids = ids_from_axtree_text(ax)
    real_labels = labels_by_id(ax)
    real_to_shown, shown_to_real = remap_ids(real_ids, mode, seed)
    shown_ax, shown_dom = _replace_current_ids(ax, dom, real_to_shown)
    shown_labels = {
        real_to_shown[real]: label
        for real, label in real_labels.items()
        if real in real_to_shown
    }
    shown_ax, action_section, stale_ids, fake_example_ids, decoy_label_ids = _with_distractors(
        shown_ax,
        action_section,
        mode,
        seed,
        shown_labels,
    )
    action_example_ids = _extract_action_example_ids(action_section)

    transformed_prompt = before_ax + shown_ax + between + shown_dom + before_action + action_section
    transform = InterfaceTransform(
        mode=mode,
        real_to_shown=real_to_shown,
        shown_to_real=shown_to_real,
        current_ids=list(shown_to_real),
        labels=shown_labels,
        stale_ids=stale_ids,
        fake_example_ids=fake_example_ids,
        action_example_ids=action_example_ids,
        decoy_label_ids=decoy_label_ids,
    )
    return transformed_prompt, transform


def unmap_action(action, transform_or_mapping):
    mapping = (
        transform_or_mapping.shown_to_real
        if isinstance(transform_or_mapping, InterfaceTransform)
        else transform_or_mapping
    )
    parsed = parse_action(action)
    if not parsed or parsed["bid"] not in mapping:
        return action
    real = mapping[parsed["bid"]]
    pattern = r"""(['"]){}(['"])""".format(re.escape(parsed["bid"]))
    return re.sub(pattern, lambda m: f"{m.group(1)}{real}{m.group(2)}", str(action), count=1)


def executable_action_from_shown(action, transform_or_mapping, invalid_id="__invalid_interface_id__"):
    """Convert a shown-interface action into a BrowserGym action.

    Invalid shown ids must not fall through as real BrowserGym bids.  Without
    this guard, a model that ignores a remapped schema and emits an original
    numeric bid can accidentally succeed in the real environment.
    """
    mapping = (
        transform_or_mapping.shown_to_real
        if isinstance(transform_or_mapping, InterfaceTransform)
        else transform_or_mapping
    )
    parsed = parse_action(action)
    if not parsed or not parsed["bid"]:
        return action
    if parsed["bid"] in mapping:
        return unmap_action(action, mapping)
    pattern = r"""(['"]){}(['"])""".format(re.escape(parsed["bid"]))
    return re.sub(pattern, lambda m: f"{m.group(1)}{invalid_id}{m.group(2)}", str(action), count=1)


def action_metrics(action, valid_ids=None, labels=None, gold_id=None, transform=None):
    parsed = parse_action(action)
    bid = parsed["bid"] if parsed else ""
    action_type = parsed["type"] if parsed else ""
    if transform is not None:
        valid_ids = transform.current_ids
        labels = transform.labels_for_metrics()
        stale_ids = set(transform.stale_ids)
        fake_example_ids = set(transform.fake_example_ids)
        action_example_ids = set(transform.action_example_ids)
        decoy_label_ids = set(transform.decoy_label_ids)
    else:
        stale_ids = set()
        fake_example_ids = set()
        action_example_ids = set()
        decoy_label_ids = set()

    valid = set(str(x) for x in (valid_ids or []))
    label_values = " ".join((labels or {}).values()).lower()
    label_tokens = set(re.findall(r"[\w@.-]+", label_values))
    id_arg_expected = action_type in ID_ARGUMENT_ACTIONS
    invalid_current_id = bool(parsed and id_arg_expected and bid and bid not in valid)
    copied_example_id = bool(parsed and bid and bid not in valid and bid in (fake_example_ids | action_example_ids))
    stale_id = bool(parsed and bid and bid in stale_ids)
    label_as_id = bool(
        parsed
        and bid
        and bid not in valid
        and (bid.lower() in label_tokens or bid in decoy_label_ids)
    )
    missing_id_arg = bool(parsed and id_arg_expected and not bid)
    bogus_argument = bool(
        (not parsed)
        or missing_id_arg
        or (
            invalid_current_id
            and not copied_example_id
            and not stale_id
            and not label_as_id
        )
    )

    return {
        "parse_valid": bool(parsed),
        "action_type": action_type,
        "action_type_valid": bool(parsed and action_type in KNOWN_ACTIONS),
        "action_bid": bid,
        "action_first_arg": bid,
        "id_argument_expected": bool(parsed and id_arg_expected),
        "valid_current_id": bool(parsed and id_arg_expected and bid in valid),
        "invalid_bid": invalid_current_id,
        "invalid_current_id": invalid_current_id,
        "copied_example_id": copied_example_id,
        "stale_id": stale_id,
        "label_as_id": label_as_id,
        "label_as_bid": label_as_id,
        "label_as_argument": label_as_id,
        "bogus_argument": bogus_argument,
        "bogus_bid": bogus_argument,
        "invalid_or_bogus_argument": bogus_argument,
        "gold_id": gold_id or "",
        "gold_id_match": bool(gold_id and bid == str(gold_id)),
    }
