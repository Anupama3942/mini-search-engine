import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from fuzzy_search import (
    levenshtein_distance, 
    levenshtein_distance_optimized,
    max_edit_distance, 
    find_fuzzy_matches, 
    resolve_term
)

class TestFuzzySearch(unittest.TestCase):
    
    def test_levenshtein_base_cases(self):
        self.assertEqual(levenshtein_distance("", ""), 0)
        self.assertEqual(levenshtein_distance("a", ""), 1)
        self.assertEqual(levenshtein_distance("", "abc"), 3)
        self.assertEqual(levenshtein_distance("python", "python"), 0)

    def test_levenshtein_insertions(self):
        # pythn -> python (insert 'o')
        self.assertEqual(levenshtein_distance("pythn", "python"), 1)
        self.assertEqual(levenshtein_distance("progrmming", "programming"), 1)

    def test_levenshtein_deletions(self):
        # programmin -> programming (1 deletion if transforming from programming to programmin)
        self.assertEqual(levenshtein_distance("programmin", "programming"), 1)
        self.assertEqual(levenshtein_distance("pythons", "python"), 1)

    def test_levenshtein_substitutions(self):
        # jython -> python ('j' replaced with 'p')
        self.assertEqual(levenshtein_distance("jython", "python"), 1)
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)

    def test_levenshtein_case_insensitivity(self):
        self.assertEqual(levenshtein_distance("PyThOn", "python"), 0)
        self.assertEqual(levenshtein_distance("PYTHON", "pythn"), 1)

    def test_optimized_levenshtein(self):
        pairs = [
            ("", ""),
            ("a", ""),
            ("", "abc"),
            ("python", "python"),
            ("pythn", "python"),
            ("programmin", "programming"),
            ("jython", "python"),
            ("kitten", "sitting")
        ]
        for s1, s2 in pairs:
            self.assertEqual(
                levenshtein_distance(s1, s2),
                levenshtein_distance_optimized(s1, s2)
            )

    def test_max_edit_distance_thresholds(self):
        self.assertEqual(max_edit_distance("cat"), 0)       # length 3 -> 0
        self.assertEqual(max_edit_distance("java"), 1)      # length 4 -> 1
        self.assertEqual(max_edit_distance("python"), 1)    # length 6 -> 1
        self.assertEqual(max_edit_distance("learning"), 2)  # length 8 -> 2

    def test_find_fuzzy_matches_ranking(self):
        vocab = {"python", "cython", "pylon", "java", "database"}
        # For 'pythn':
        # python (dist 1)
        # cython (dist 2 - but length 5 allows max_dist 1, so filtered out!)
        # pylon (dist 2 - filtered out)
        matches = find_fuzzy_matches("pythn", vocab)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["term"], "python")
        self.assertEqual(matches[0]["distance"], 1)

    def test_resolve_term_exact_vs_fuzzy(self):
        vocab = {"python", "programming", "java", "database"}
        cache = {}

        # Exact match
        resolved, is_fuzzy, dist = resolve_term("python", vocab, cache)
        self.assertEqual(resolved, "python")
        self.assertFalse(is_fuzzy)
        self.assertEqual(dist, 0)

        # Fuzzy match
        resolved, is_fuzzy, dist = resolve_term("pythn", vocab, cache)
        self.assertEqual(resolved, "python")
        self.assertTrue(is_fuzzy)
        self.assertEqual(dist, 1)

        # Cached lookup
        self.assertIn("pythn", cache)
        resolved_cached, is_fuzzy_c, dist_c = resolve_term("pythn", vocab, cache)
        self.assertEqual(resolved_cached, "python")

        # Unknown term with no close match
        resolved_unk, is_fuzzy_u, _ = resolve_term("qwertyuiopasdf", vocab, cache)
        self.assertEqual(resolved_unk, "qwertyuiopasdf")
        self.assertFalse(is_fuzzy_u)

if __name__ == '__main__':
    unittest.main()
