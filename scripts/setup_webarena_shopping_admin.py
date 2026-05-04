#!/usr/bin/env python3
"""Download and start the WebArena shopping-admin backend."""

import argparse
import concurrent.futures
import os
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen


URL = "http://metis.lti.cs.cmu.edu/webarena-images/shopping_admin_final_0719.tar"
SIZE = 9_640_032_256
IMAGE = "shopping_admin_final_0719"


def run(cmd, **kwargs):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def download_part(args):
    idx, start, end, url, parts_dir = args
    expected = end - start + 1
    final = parts_dir / f"part_{idx:04d}"
    if final.exists() and final.stat().st_size == expected:
        return idx, expected, "cached"

    tmp = final.with_suffix(".tmp")
    req = Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urlopen(req, timeout=120) as resp, tmp.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    got = tmp.stat().st_size
    if got != expected:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"part {idx} size {got}, expected {expected}")
    tmp.replace(final)
    return idx, expected, "downloaded"


def download_tar(out_dir, workers, chunk_mb):
    out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = out_dir / ".shopping_admin_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / "shopping_admin_final_0719.tar"
    if tar_path.exists() and tar_path.stat().st_size == SIZE:
        print(f"Using existing {tar_path}", flush=True)
        return tar_path

    chunk = chunk_mb * 1024 * 1024
    jobs = []
    for idx, start in enumerate(range(0, SIZE, chunk)):
        end = min(start + chunk - 1, SIZE - 1)
        jobs.append((idx, start, end, URL, parts_dir))

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(download_part, job) for job in jobs]
        for fut in concurrent.futures.as_completed(futures):
            idx, size, status = fut.result()
            done += size
            print(f"part {idx:04d} {status}; {done / SIZE:.1%}", flush=True)

    tmp_tar = tar_path.with_suffix(".tar.tmp")
    with tmp_tar.open("wb") as out:
        for idx in range(len(jobs)):
            part = parts_dir / f"part_{idx:04d}"
            if not part.exists():
                raise RuntimeError(f"missing {part}")
            with part.open("rb") as f:
                while True:
                    chunk_data = f.read(16 * 1024 * 1024)
                    if not chunk_data:
                        break
                    out.write(chunk_data)

    got = tmp_tar.stat().st_size
    if got != SIZE:
        raise RuntimeError(f"final tar size {got}, expected {SIZE}")
    tmp_tar.replace(tar_path)
    return tar_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="/mnt/code/webarena_images")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-mb", type=int, default=128)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="7780")
    args = parser.parse_args()

    tar_path = download_tar(Path(args.out_dir), args.workers, args.chunk_mb)
    run(["docker", "load", "--input", str(tar_path)])
    run(["docker", "rm", "-f", "shopping_admin"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run(["docker", "run", "--name", "shopping_admin", "-p", f"{args.port}:80", "-d", IMAGE])
    time.sleep(75)

    base_url = f"http://{args.host}:{args.port}"
    run([
        "docker",
        "exec",
        "shopping_admin",
        "/var/www/magento2/bin/magento",
        "setup:store-config:set",
        f"--base-url={base_url}",
    ])
    run([
        "docker",
        "exec",
        "shopping_admin",
        "mysql",
        "-u",
        "magentouser",
        "-pMyPassword",
        "magentodb",
        "-e",
        f'UPDATE core_config_data SET value="{base_url}/" WHERE path = "web/secure/base_url";',
    ])
    for key, value in [
        ("admin/security/password_is_forced", "0"),
        ("admin/security/password_lifetime", "0"),
    ]:
        run([
            "docker",
            "exec",
            "shopping_admin",
            "php",
            "/var/www/magento2/bin/magento",
            "config:set",
            key,
            value,
        ])
    run(["docker", "exec", "shopping_admin", "/var/www/magento2/bin/magento", "cache:flush"])
    run(["curl", "-s", "-o", "/dev/null", "-w", "shopping_admin %{http_code}\n", f"{base_url}/admin"])


if __name__ == "__main__":
    main()
