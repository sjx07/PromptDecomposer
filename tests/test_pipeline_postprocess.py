import unittest

from PromptDecomposer.align import _tokenize
from PromptDecomposer.pipeline import PromptDecomposer
from PromptDecomposer.structure import build_structure_candidates
from PromptDecomposer.utils import split_units


class PipelinePostprocessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.decomposer = PromptDecomposer()

    def test_hoists_single_prompt_heading_wrapper(self) -> None:
        prompt = "## Talking to the user\nFirst block.\n\nSecond block."
        tree = [
            {
                "id": "old-root",
                "type": "talking_to_the_user",
                "span": [0, len(prompt)],
                "metadata": {},
                "children": [
                    {
                        "id": "old-a",
                        "type": "visibility",
                        "span": [23, 35],
                        "metadata": {},
                        "children": [],
                    },
                    {
                        "id": "old-b",
                        "type": "timing",
                        "span": [37, len(prompt)],
                        "metadata": {},
                        "children": [],
                    },
                ],
            }
        ]

        result = self.decomposer._postprocess_tree("p0", prompt, tree)

        self.assertEqual([node["type"] for node in result], ["visibility", "timing"])
        self.assertEqual(result[0]["span"], [23, 35])
        self.assertEqual(result[0]["metadata"]["title"], "## Talking to the user")
        self.assertEqual(result[0]["metadata"]["parent_id"], None)
        self.assertEqual(result[0]["metadata"]["depth"], 0)
        self.assertEqual(
            result[0]["metadata"]["postprocess_actions"][-1]["action"],
            "hoisted_prompt_wrapper",
        )

    def test_merges_header_child_and_collapses_wrapper(self) -> None:
        prompt = "Usage:\n- Rule A\n- Rule B"
        tree = [
            {
                "id": "old-root",
                "type": "usage_guidelines",
                "span": [0, len(prompt)],
                "metadata": {},
                "children": [
                    {
                        "id": "old-heading",
                        "type": "usage_header",
                        "span": [0, 6],
                        "metadata": {},
                        "children": [],
                    },
                    {
                        "id": "old-rules",
                        "type": "usage_rules",
                        "span": [7, len(prompt)],
                        "metadata": {},
                        "children": [],
                    },
                ],
            }
        ]

        result = self.decomposer._postprocess_tree("p0", prompt, tree)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "usage_rules")
        self.assertEqual(result[0]["span"], [7, len(prompt)])
        self.assertEqual(result[0]["metadata"]["title"], "Usage:")
        actions = [
            action["action"]
            for action in result[0]["metadata"]["postprocess_actions"]
        ]
        self.assertIn("collapsed_single_child_wrapper", actions)

    def test_promotes_parent_heading_without_expanding_first_child(self) -> None:
        prompt = "### Prompt tips\n**Good examples:**\n1. A\n2. B"
        tree = [
            {
                "id": "old-root",
                "type": "prompt_tips",
                "span": [0, len(prompt)],
                "metadata": {},
                "children": [
                    {
                        "id": "old-heading",
                        "type": "section_heading",
                        "span": [0, 15],
                        "metadata": {},
                        "children": [],
                    },
                    {
                        "id": "old-good",
                        "type": "good_examples_list",
                        "span": [16, len(prompt)],
                        "metadata": {},
                        "children": [],
                    },
                ],
            }
        ]

        result = self.decomposer._postprocess_tree("p0", prompt, tree)

        self.assertEqual(result[0]["metadata"]["title"], "### Prompt tips")
        self.assertEqual(result[0]["span"], [16, len(prompt)])
        self.assertNotEqual(result[0]["span"][0], 0)

    def test_promotes_child_subheader_without_expanding_first_item(self) -> None:
        prompt = "**Good examples:**\n1. A\n2. B"
        tree = [
            {
                "id": "old-root",
                "type": "good_examples_list",
                "span": [0, len(prompt)],
                "metadata": {},
                "children": [
                    {
                        "id": "old-heading",
                        "type": "list_heading",
                        "span": [0, 18],
                        "metadata": {},
                        "children": [],
                    },
                    {
                        "id": "old-a",
                        "type": "example_item",
                        "span": [19, 23],
                        "metadata": {},
                        "children": [],
                    },
                    {
                        "id": "old-b",
                        "type": "example_item",
                        "span": [24, len(prompt)],
                        "metadata": {},
                        "children": [],
                    },
                ],
            }
        ]

        result = self.decomposer._postprocess_tree("p0", prompt, tree)

        self.assertEqual(result[0]["metadata"]["title"], "**Good examples:**")
        self.assertEqual(result[0]["children"][0]["span"], [19, 23])

    def test_marks_same_type_list_items_structurally(self) -> None:
        prompt = "Rules:\n- Rule A\n- Rule B"
        tree = [
            {
                "id": "old-root",
                "type": "rules",
                "span": [0, len(prompt)],
                "metadata": {},
                "children": [
                    {
                        "id": "old-a",
                        "type": "rules",
                        "span": [7, 15],
                        "metadata": {},
                        "children": [],
                    },
                    {
                        "id": "old-b",
                        "type": "rules",
                        "span": [16, len(prompt)],
                        "metadata": {},
                        "children": [],
                    },
                ],
            }
        ]

        result = self.decomposer._postprocess_tree("p0", prompt, tree)
        children = result[0]["children"]

        self.assertEqual(children[0]["metadata"]["structural_role"], "list_item")
        self.assertEqual(children[1]["metadata"]["structural_role"], "list_item")

    def test_does_not_collapse_meaningful_prose_prefix(self) -> None:
        prefix = "You are a coordinator.\n"
        prompt = f"{prefix}- Rule A\n- Rule B"
        tree = [
            {
                "id": "old-root",
                "type": "coordinator_guidance",
                "span": [0, len(prompt)],
                "metadata": {},
                "children": [
                    {
                        "id": "old-rules",
                        "type": "rules",
                        "span": [len(prefix), len(prompt)],
                        "metadata": {},
                        "children": [],
                    },
                ],
            }
        ]

        result = self.decomposer._postprocess_tree("p0", prompt, tree)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "coordinator_guidance")
        self.assertEqual(result[0]["span"], [0, len(prompt)])
        self.assertEqual(result[0]["children"][0]["span"], [len(prefix), len(prompt)])
        actions = result[0]["children"][0].get("metadata", {}).get("postprocess_actions", [])
        self.assertNotIn("collapsed_single_child_wrapper", [a["action"] for a in actions])

    def test_align_segments_fills_missing_structural_candidates(self) -> None:
        prompt = "\n".join([
            "Intro.",
            "## Rules",
            "- A",
            "## Output",
            "JSON only.",
        ])
        units, unit_spans = split_units(prompt)
        unit_tokens = [_tokenize(unit) for unit in units]
        candidates = build_structure_candidates(units)

        aligned = self.decomposer._align_segments(
            units,
            unit_tokens,
            unit_spans,
            [
                {
                    "span_id": "S1",
                    "label": "rules",
                    "content": "## Rules\n- A",
                    "reason": "candidate label",
                    "confidence": "high",
                    "boundary_cues": ["markdown_heading"],
                    "anchor_phrases": ["## Rules"],
                    "should_refine": True,
                }
            ],
            reject_ambiguous=False,
            structural_candidates=candidates,
        )

        self.assertEqual([record["unit_range"] for record in aligned], [[0, 0], [1, 2], [3, 4]])
        self.assertEqual(aligned[1]["type"], "rules")
        self.assertEqual(aligned[1]["structural_candidate"]["id"], "S1")
        self.assertEqual(aligned[2]["structural_candidate"]["kind"], "markdown_section")

    def test_align_segments_without_structure_uses_content_alignment(self) -> None:
        prompt = "Do the requested work.\nExplain the result clearly."
        units, unit_spans = split_units(prompt)
        unit_tokens = [_tokenize(unit) for unit in units]
        candidates = build_structure_candidates(units)

        aligned = self.decomposer._align_segments(
            units,
            unit_tokens,
            unit_spans,
            [
                {
                    "label": "task",
                    "content": "Do the requested work.",
                    "reason": "first sentence",
                    "confidence": "high",
                    "boundary_cues": ["paragraph"],
                    "anchor_phrases": ["requested work"],
                    "should_refine": False,
                },
                {
                    "label": "style_constraints",
                    "content": "Explain the result clearly.",
                    "reason": "second sentence",
                    "confidence": "high",
                    "boundary_cues": ["paragraph"],
                    "anchor_phrases": ["result clearly"],
                    "should_refine": False,
                },
            ],
            reject_ambiguous=False,
            structural_candidates=candidates,
        )

        self.assertEqual(candidates, [])
        self.assertEqual([record["unit_range"] for record in aligned], [[0, 0], [1, 1]])
        self.assertEqual([record["type"] for record in aligned], ["task", "style_constraints"])
        self.assertNotIn("structural_candidate", aligned[0])

    def test_atomizes_rule_list_even_when_semantically_coherent(self) -> None:
        self.decomposer.mode = "free"
        prompt = "Intro.\nRules:\n- Do A\n- Do B"
        units, unit_spans = split_units(prompt)
        unit_tokens = [_tokenize(unit) for unit in units]
        self.decomposer._request_segments = lambda *args, **kwargs: [
            {
                "span_id": "S0",
                "label": "intro",
                "content": "Intro.",
                "should_refine": False,
            },
            {
                "span_id": "S1",
                "label": "rules",
                "content": "Rules:\n- Do A\n- Do B",
                "should_refine": False,
            },
        ]
        self.decomposer._request_atomize = lambda _section_text: [
            {"content": "- Do A", "kind": "requirement", "confidence": "high"},
            {"content": "- Do B", "kind": "requirement", "confidence": "high"},
        ]

        tree = self.decomposer._decompose_scope(
            "p0",
            prompt,
            units,
            unit_tokens,
            unit_spans,
            depth=0,
            parent_id=None,
            parent_type=None,
            top_level=True,
        )

        rules = tree[1]
        self.assertEqual(rules["type"], "rules")
        self.assertEqual([child["kind"] for child in rules["children"]], ["requirement", "requirement"])
        self.assertEqual([prompt[c["span"][0]:c["span"][1]] for c in rules["children"]], ["- Do A", "- Do B"])

    def test_keeps_examples_list_coherent(self) -> None:
        self.decomposer.mode = "free"
        prompt = "Good examples:\n1. Do A well\n2. Do B well\nFinal."
        units, unit_spans = split_units(prompt)
        unit_tokens = [_tokenize(unit) for unit in units]
        self.decomposer._request_segments = lambda *args, **kwargs: [
            {
                "span_id": "S0",
                "label": "good_examples",
                "content": "Good examples:\n1. Do A well\n2. Do B well",
                "should_refine": True,
            },
            {
                "span_id": "S1",
                "label": "final_note",
                "content": "Final.",
                "should_refine": False,
            },
        ]
        self.decomposer._request_atomize = lambda _section_text: self.fail("examples should stay coherent")

        tree = self.decomposer._decompose_scope(
            "p0",
            prompt,
            units,
            unit_tokens,
            unit_spans,
            depth=0,
            parent_id=None,
            parent_type=None,
            top_level=True,
        )

        self.assertEqual(tree[0]["type"], "good_examples")
        self.assertEqual(tree[0]["children"], [])


if __name__ == "__main__":
    unittest.main()
