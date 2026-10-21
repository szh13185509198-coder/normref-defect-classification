"""Vendored MVREC upstream completeness regression tests.

The vendored ``MVREC-main/`` must be a faithful copy of the locked official
upstream commit ``112c187b6de8a7400fad89ca2a201545842a5d84``.  An earlier
commit silently dropped ``MVREC-main/lyus/data/`` because the repository
``.gitignore`` rule ``data/`` matched that vendored SOURCE directory as if
it were a dataset directory, which broke ``import lyus.data.my_transforms``
in the cloud.  These tests fail-closed on that regression:

- every locked ``lyus/data`` file must exist;
- every file SHA256 must equal the value pinned from the official upstream
  commit (verified against ``git show 112c187b:lyus/data/<file>``);
- key vendored modules outside ``lyus/data`` must also exist.

File-level checks only: nothing here imports the vendored code, so the
test runs without any model/dependency side effects.
"""

import hashlib
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MVREC_MAIN = PROJECT_ROOT / "MVREC-main"

MVREC_UPSTREAM_COMMIT = "112c187b6de8a7400fad89ca2a201545842a5d84"

# SHA256 of every lyus/data file at the locked upstream commit.
LYUS_DATA_SHA256 = {
    "__init__.py": "c1b5a1c5703d6124bfb7eaa21a507545d7c5b8d0f7a54dec702d1196ad3c391f",
    "base_dataset.py": "a0ef31e63b7647c30fb7824c348e92f9afea8ccd27894917583c4e16e5c53e59",
    "csv_data.py": "05634b245075949d638c7840a9fbe83e54182676d369c112c81a2f20632fc16d",
    "data_manager.py": "40a060c94086b8a90fd9748215dad365eba12ac4a5441e0e743054e08aade171",
    "data_managers.py": "a19f85dcfd2be4b1b4026381ecc705cb9aa39b13f89491789c69575fee989597",
    "folder_dataset.py": "75c03c14b198663b92808acf16d28826bd78439d827212307f0cd2745deb53e1",
    "gen_defect.py": "e202e89fb4ef1768e82cae0b3d8591ed5c2b4d00070c321219b0839ac49d15d0",
    "jsontool.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "my_transforms.py": "5294acacd70619d7925d15b9b38042376eb7947650e8a81e47b5b7fa211af1cc",
    "patch_data.py": "1ab2e3a4fbf5aca360a78d020915885efe7844662086c9a6ad5c0b86111d864f",
    "sample_process.py": "abdfa22dab11d884ceee12552087916488953d84b2fffe034ae53b494860ee5f",
    "sampler.py": "3f9a1911a1fd5c1c55d2718d4a47ce1840c40ed748d29d23363c821c61a1ce65",
    "utils.py": "04c94ede30012fac08b54a30b701d5befd94be4026333e7585d484e657118767",
}

# Other key vendored files whose presence guards the vendor copy.
KEY_VENDORED_FILES = (
    "run.py",
    "fewshot_process.py",
    "data_param.py",
    "param_space.py",
    "modules/model.py",
    "modules/classifier.py",
    "modules/classifier_base.py",
    "modules/clip_preprocess.py",
    "modules/fabric_data.py",
    "modules/multi_view_data.py",
    "modules/part_data.py",
    "modules/torch_demo.py",
    "modules/utils.py",
    "modules/bbox_clip/alpha_clip_backbone.py",
    "lyus/Frame/Exp/experiment.py",
    "lyus/Frame/Exp/param.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TestVendoredUpstreamCompleteness(unittest.TestCase):
    def test_01_lyus_data_directory_present(self):
        self.assertTrue((MVREC_MAIN / "lyus" / "data").is_dir(),
                        "MVREC-main/lyus/data/ is missing from the vendor")

    def test_02_every_lyus_data_file_exists_with_locked_sha(self):
        missing = []
        mismatched = []
        for filename, expected in LYUS_DATA_SHA256.items():
            path = MVREC_MAIN / "lyus" / "data" / filename
            if not path.is_file():
                missing.append(filename)
                continue
            actual = _sha256(path)
            if actual != expected:
                mismatched.append((filename, actual))
        self.assertEqual(missing, [],
                         f"missing lyus/data files: {missing}")
        self.assertEqual(mismatched, [],
                         f"lyus/data SHA drift vs {MVREC_UPSTREAM_COMMIT}: "
                         f"{mismatched}")

    def test_03_my_transforms_importable_name_present(self):
        path = MVREC_MAIN / "lyus" / "data" / "my_transforms.py"
        self.assertTrue(path.is_file(),
                        "my_transforms.py missing; cloud import "
                        "'lyus.data.my_transforms' would fail")

    def test_04_key_vendored_files_present(self):
        missing = [rel for rel in KEY_VENDORED_FILES
                   if not (MVREC_MAIN.joinpath(*rel.split("/"))).is_file()]
        self.assertEqual(missing, [], f"missing vendored files: {missing}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
