import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interface_variants import (  # noqa: E402
    INTERFACE_MODES,
    action_metrics,
    apply_interface_variant,
    executable_action_from_shown,
    parse_interface_modes,
    unmap_action,
)


PROMPT = """# Instructions

Act.

# Goal
Click submit.

# Currently open tabs
Tab 0 (active tab)
  Title: test
  URL: test

# Current page Accessibility Tree
[12] button 'Submit'
[13] textbox 'Name'
[14] combobox 'Choice'

# Current page DOM
<button bid="12" data-bid='12'>Submit</button>
<input bid="13" data-bid='13' />
<select bid="14"><option>A</option></select>

# Action Space

click('48')
fill('49', 'text')
select_option('50', 'A')

# Error message from last action


# Next action
Respond with one action."""


class InterfaceVariantTests(unittest.TestCase):
    def test_all_interface_modes_are_reversible_and_unique(self):
        for mode in INTERFACE_MODES:
            with self.subTest(mode=mode):
                shown_prompt, transform = apply_interface_variant(PROMPT, mode, seed=7)
                self.assertEqual(len(transform.current_ids), 3)
                self.assertEqual(len(set(transform.current_ids)), 3)
                self.assertEqual(set(transform.shown_to_real.values()), {"12", "13", "14"})

                shown_id = transform.real_to_shown["12"]
                self.assertIn(f"[{shown_id}]", shown_prompt)
                self.assertEqual(unmap_action(f"click('{shown_id}')", transform), "click('12')")

    def test_stale_and_fake_ids_are_detected_but_not_valid(self):
        _prompt, stale = apply_interface_variant(PROMPT, "stale_ids", seed=11)
        stale_id = sorted(stale.stale_ids)[0]
        stale_metrics = action_metrics(f"click('{stale_id}')", transform=stale)
        self.assertTrue(stale_metrics["parse_valid"])
        self.assertTrue(stale_metrics["stale_id"])
        self.assertFalse(stale_metrics["valid_current_id"])

        _prompt, fake = apply_interface_variant(PROMPT, "fake_examples", seed=11)
        fake_id = sorted(fake.fake_example_ids)[0]
        fake_metrics = action_metrics(f"click('{fake_id}')", transform=fake)
        self.assertTrue(fake_metrics["copied_example_id"])
        self.assertFalse(fake_metrics["valid_current_id"])

    def test_decoy_labels_do_not_become_valid_ids(self):
        _prompt, transform = apply_interface_variant(PROMPT, "decoy_labels", seed=5)
        decoy = sorted(transform.decoy_label_ids)[0]
        self.assertNotIn(decoy, transform.current_ids)
        metrics = action_metrics(f"click('{decoy}')", transform=transform)
        self.assertTrue(metrics["label_as_id"])
        self.assertFalse(metrics["valid_current_id"])

    def test_interface_mode_aliases(self):
        self.assertEqual(
            parse_interface_modes("original,alnum,stale"),
            ["original", "alphanumeric", "stale_ids"],
        )

    def test_invalid_shown_id_cannot_execute_as_real_bid(self):
        _prompt, transform = apply_interface_variant(PROMPT, "structured", seed=7)
        self.assertNotIn("12", transform.current_ids)
        executable = executable_action_from_shown("click('12')", transform)
        self.assertEqual(executable, "click('__invalid_interface_id__')")


if __name__ == "__main__":
    unittest.main()
