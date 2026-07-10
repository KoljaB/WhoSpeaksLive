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
    def test_windows_dockerfile_openssl_patch_runs_after_crlf_patch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            dockerfile = repo / "Dockerfile.windows"
            dockerfile.write_text(
                "# Windows-native OpenSSL\n"
                "RUN apt-get update && apt-get install -y --no-install-recommends \\\n"
                "        msitools innoextract \\\n"
                " && rm -rf /var/lib/apt/lists/* \\\n"
                " && mkdir -p /tmp/openssl-msi /opt/openssl-win64/app/bin \\\n"
                "        /opt/openssl-win64/app/lib /opt/openssl-win64/app/include \\\n"
                " && cd /tmp \\\n"
                " && for v in 3_6_2 3_6_1 3_5_4 3_5_3; do \\\n"
                "        if curl -sLf \"https://slproweb.com/download/Win64OpenSSL-${v}.msi\" \\\n"
                "                -o openssl.msi; then \\\n"
                "            echo \"Downloaded Win64OpenSSL-${v}.msi\"; \\\n"
                "            break; \\\n"
                "        fi; \\\n"
                "        rm -f openssl.msi; \\\n"
                "    done \\\n"
                " && test -s openssl.msi \\\n"
                " && msiextract -C /tmp/openssl-msi openssl.msi \\\n"
                " && mkdir -p /tmp/openssl-final \\\n"
                " && (find /tmp/openssl-msi -name \"*.exe\" \\\n"
                "        -exec innoextract -d /tmp/openssl-final {} + 2>/dev/null \\\n"
                "     || find /tmp/openssl-msi -name \"*.exe\" \\\n"
                "        -exec 7z x -o/tmp/openssl-final {} \\;) \\\n"
                " && (cp -r /tmp/openssl-final/app/* /opt/openssl-win64/app/ 2>/dev/null \\\n"
                "     || (find /tmp/openssl-final -name \"libcrypto*.dll\" \\\n"
                "            -exec cp -v {} /opt/openssl-win64/app/bin/ \\;)) \\\n"
                " && test -f /opt/openssl-win64/app/lib/libcrypto.lib \\\n"
                "        -o -f /opt/openssl-win64/app/lib/libcrypto_static.lib \\\n"
                " && rm -rf /tmp/openssl-msi /tmp/openssl-final /tmp/openssl.msi\n"
                "ENV OPENSSL_ROOT_DIR=/opt/openssl-win64/app\n"
                "COPY in_windows_container.sh /usr/local/bin/in_windows_container.sh\n"
                "RUN sed -i 's/\\r$//' /usr/local/bin/in_windows_container.sh \\\n"
                " && chmod +x /usr/local/bin/in_windows_container.sh\n",
                encoding="utf-8",
            )

            install_kroko.patch_windows_dockerfile(repo)
            first_text = dockerfile.read_text(encoding="utf-8")
            install_kroko.patch_windows_dockerfile(repo)
            second_text = dockerfile.read_text(encoding="utf-8")

            self.assertEqual(first_text, second_text)
            self.assertIn("WhoSpeaks patch: robust OpenSSL extraction", first_text)
            self.assertIn("/tmp/openssl-msi /tmp/openssl-final /tmp/openssl-final/app", first_text)
            self.assertIn("libcrypto*.lib", first_text)
            self.assertIn("include/openssl/ssl.h", first_text)
            self.assertIn("sed -i 's/\\r$//'", first_text)

    def test_windows_free_dockerfile_skips_openssl_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            dockerfile = repo / "Dockerfile.windows"
            dockerfile.write_text(
                "# Windows-native OpenSSL\n"
                "RUN apt-get update && apt-get install -y --no-install-recommends \\\n"
                "        msitools innoextract \\\n"
                " && rm -rf /var/lib/apt/lists/* \\\n"
                " && mkdir -p /tmp/openssl-msi /opt/openssl-win64/app/bin \\\n"
                "        /opt/openssl-win64/app/lib /opt/openssl-win64/app/include \\\n"
                " && cd /tmp \\\n"
                " && for v in 3_6_2 3_6_1 3_5_4 3_5_3; do \\\n"
                "        if curl -sLf \"https://slproweb.com/download/Win64OpenSSL-${v}.msi\" \\\n"
                "                -o openssl.msi; then \\\n"
                "            break; \\\n"
                "        fi; \\\n"
                "    done \\\n"
                " && test -s openssl.msi \\\n"
                " && msiextract -C /tmp/openssl-msi openssl.msi \\\n"
                " && mkdir -p /tmp/openssl-final \\\n"
                " && test -f /opt/openssl-win64/app/lib/libcrypto.lib \\\n"
                " && rm -rf /tmp/openssl-msi /tmp/openssl-final /tmp/openssl.msi\n"
                "ENV OPENSSL_ROOT_DIR=/opt/openssl-win64/app\n"
                "COPY in_windows_container.sh /usr/local/bin/in_windows_container.sh\n"
                "RUN chmod +x /usr/local/bin/in_windows_container.sh\n",
                encoding="utf-8",
            )

            install_kroko.patch_windows_dockerfile(repo, install_openssl=False)
            first_text = dockerfile.read_text(encoding="utf-8")
            install_kroko.patch_windows_dockerfile(repo, install_openssl=False)
            second_text = dockerfile.read_text(encoding="utf-8")

            self.assertEqual(first_text, second_text)
            self.assertIn("free Windows build skips OpenSSL download", first_text)
            self.assertNotIn("slproweb.com", first_text)
            self.assertNotIn("msiextract", first_text)
            self.assertIn("/opt/openssl-win64/app/bin", first_text)
            self.assertIn("sed -i 's/\\r$//'", first_text)

    def test_windows_free_build_is_wheel_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            script = repo / "in_windows_container.sh"
            batch = repo / "build_windows.bat"
            script.write_text(
                "#!/bin/bash\n"
                "BUILD_VARIANT=\"${BUILD_VARIANT:-pro}\"\n"
                "case \"$BUILD_VARIANT\" in\n"
                "    pro)  KROKO_LICENSE_FLAG=ON ;;\n"
                "    free) KROKO_LICENSE_FLAG=OFF ;;\n"
                "esac\n"
                "echo \"    Variant: $BUILD_VARIANT (KROKO_LICENSE=$KROKO_LICENSE_FLAG)\"\n"
                "echo\n"
                "cd /src\n"
                "BUILD_DIR=/tmp/build-windows\n"
                "INSTALL_DIR=/out\n"
                "# Configure. The websocket server is gated by SHERPA_ONNX_ENABLE_WEBSOCKET\n"
                "cmake \\\n"
                "    -DSHERPA_ONNX_ENABLE_WEBSOCKET=ON \\\n"
                "    -DKROKO_LICENSE=$KROKO_LICENSE_FLAG \\\n"
                "    \"$@\"\n"
                "cmake --build \"$BUILD_DIR\" \\\n"
                "    --target kroko-onnx-online-websocket-server \\\n"
                "    --parallel \"$(nproc)\"\n"
                "mkdir -p \"$INSTALL_DIR/bin\"\n"
                "if [ ! -f \"$INSTALL_DIR/bin/kroko-onnx-online-websocket-server.exe\" ]; then\n"
                "    echo \"ERROR: kroko-onnx-online-websocket-server.exe missing from build output\" >&2\n"
                "    exit 1\n"
                "fi\n"
                "\n"
                "# Python wheel\n"
                "export SHERPA_ONNX_CMAKE_ARGS=\" \\\n"
                "    -DSHERPA_ONNX_ENABLE_BINARY=OFF \\\n"
                "    -DSHERPA_ONNX_ENABLE_WEBSOCKET=ON \\\n"
                "    -DKROKO_LICENSE=$KROKO_LICENSE_FLAG\"\n",
                encoding="utf-8",
            )
            batch.write_text(
                "@echo off\r\n"
                ":build_variant\r\n"
                "set \"VARIANT=%~1\"\r\n"
                "if exist \"%HOST_OUT%\\wheel\" (\r\n"
                "    echo Wheel\r\n"
                ")\r\n"
                "rmdir /S /Q \"%HOST_OUT%\" 2>nul\r\n"
                "\r\n"
                "REM -- Step 3: build the NSIS installer ---------------------------------------\r\n"
                "echo [3/3] Building NSIS installer (%VARIANT%)\r\n",
                encoding="utf-8",
            )

            install_kroko.patch_windows_free_wheel_only_build(repo)
            first_script = script.read_text(encoding="utf-8")
            first_batch = batch.read_text(encoding="utf-8")
            install_kroko.patch_windows_free_wheel_only_build(repo)
            second_script = script.read_text(encoding="utf-8")
            second_batch = batch.read_text(encoding="utf-8")

            self.assertEqual(first_script, second_script)
            self.assertEqual(first_batch, second_batch)
            self.assertIn("WhoSpeaks patch: free Windows build is wheel-only", first_script)
            self.assertIn("KROKO_WHEEL_ONLY=1", first_script)
            self.assertIn("KROKO_WEBSOCKET_FLAG=OFF", first_script)
            self.assertIn('if [ "$KROKO_WHEEL_ONLY" = "1" ]; then', first_script)
            self.assertIn("Skipping Windows websocket-server build for free wheel-only runtime.", first_script)
            self.assertNotIn("-DSHERPA_ONNX_ENABLE_WEBSOCKET=ON", first_script)
            self.assertIn("-DSHERPA_ONNX_ENABLE_WEBSOCKET=$KROKO_WEBSOCKET_FLAG", first_script)
            self.assertIn("WhoSpeaks patch: skip NSIS for free wheel-only build", first_batch)
            self.assertIn('if /I "%VARIANT%"=="free"', first_batch)

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
