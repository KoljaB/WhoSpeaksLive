from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from RealtimeSTT import install_kroko


class KrokoInstallerTests(unittest.TestCase):
    def test_free_linux_patch_makes_openssl_license_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            csrc = repo / "sherpa-onnx" / "csrc"
            cmake = repo / "cmake"
            csrc.mkdir(parents=True)
            cmake.mkdir()

            (csrc / "CMakeLists.txt").write_text(
                "include_directories(${PROJECT_SOURCE_DIR})\n\n"
                "find_package(OpenSSL REQUIRED)\n\n"
                "target_link_libraries(kroko-onnx-core\n"
                "  kaldi-native-fbank-core\n"
                "  kaldi-decoder-core\n"
                "  ssentencepiece_core\n"
                "  OpenSSL::SSL\n"
                "  OpenSSL::Crypto\n"
                ")\n",
                encoding="utf-8",
            )
            (csrc / "online-transducer-model.cc").write_text(
                '#include "sherpa-onnx/csrc/ModelData.h"\n'
                '#include "sherpa-onnx/csrc/license.h"\n',
                encoding="utf-8",
            )
            (csrc / "ModelData.cc").write_text(
                '#include "sherpa-onnx/csrc/ModelData.h"\n'
                "#include <openssl/aes.h>\n"
                "#include <openssl/evp.h>\n"
                "#include <openssl/rand.h>\n\n"
                "bool ModelData::decryptPayload(const std::string& password) {\n"
                "    return !password.empty();\n"
                "}\n\n"
                "bool ModelData::loadPayload() {\n"
                "    return true;\n"
                "}\n",
                encoding="utf-8",
            )
            (cmake / "cmake_extension.py").write_text(
                "import os\n\n"
                "def get_binaries():\n"
                "    binaries = [\n"
                '        "sherpa-onnx-offline-websocket-server",\n'
                '        "kroko-onnx-online-websocket-server",\n'
                '        "sherpa-onnx-version",\n'
                "    ]\n\n"
                "    if enable_alsa():\n"
                "        binaries += [\n"
                '            "sherpa-onnx-alsa",\n'
                "        ]\n"
                "    return binaries\n",
                encoding="utf-8",
            )

            install_kroko.patch_linux_free_build_without_openssl_dev(repo)

            cmake_text = (csrc / "CMakeLists.txt").read_text(encoding="utf-8")
            self.assertIn("if(KROKO_LICENSE)\n  find_package(OpenSSL REQUIRED)\nendif()", cmake_text)
            self.assertIn("if(KROKO_LICENSE)\n  target_link_libraries", cmake_text)
            transducer_text = (csrc / "online-transducer-model.cc").read_text(encoding="utf-8")
            self.assertIn("#ifdef KROKO_LICENSE\n#include \"sherpa-onnx/csrc/license.h\"\n#endif", transducer_text)
            model_text = (csrc / "ModelData.cc").read_text(encoding="utf-8")
            self.assertIn("#ifdef KROKO_LICENSE\n#include <openssl/aes.h>", model_text)
            self.assertIn("#else\nbool ModelData::decryptPayload(const std::string&) {", model_text)
            extension_text = (cmake / "cmake_extension.py").read_text(encoding="utf-8")
            self.assertIn("SHERPA_ONNX_ENABLE_WEBSOCKET", extension_text)
            self.assertIn('if "websocket" not in item', extension_text)


if __name__ == "__main__":
    unittest.main()
