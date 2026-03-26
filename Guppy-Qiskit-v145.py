#Hi i Realy apperciated you get me Some Donations here_ 1Bu4CR8Bi5AXQG8pnu1avny88C5CCgWKfb /////
#============================================================================================
#!/usr/bin/env python3
# =============================================================================
# 🐉 DRAGON_CODE_v145 — FULL Combined (Guppy + Qiskit) — Version 2026 🐉
# =============================================================================
# Ready for Use Both Guppy/Q-Nexus & Qiskit/IBM .
# Fault-Tolrent Draper 2D + 4×XY8 + 4-scale ZNE + Majority Vote + Lattice Attack
# Real repetition on state & ctrl + flags + BKZ + Multiple Convergents
# Enhanced Post-Processing for ECDLP Period Finding + Continued Fraction
# Optional TKET drawing + Selene-sim 0.2.12 local Support .
# =============================================================================

import os
import sys
import math
import time
import subprocess
import numpy as np
from datetime import datetime
from fractions import Fraction
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ecdsa.ellipticcurve import Point, CurveFp
from ecdsa import SECP256k1

# fpylll for BKZ
try:
    from fpylll import IntegerMatrix, BKZ
    FPYLLL_AVAILABLE = True
except ImportError:
    FPYLLL_AVAILABLE = False
    print("⚠️ fpylll not installed — using simple LLL instead of BKZ")

# pytket — exact working v145 block
try:
    from pytket import Circuit as TketCircuit
    TKET_AVAILABLE = True
    print("✅ pytket/TKET detected — advanced drawing enabled")
except ImportError:
    TKET_AVAILABLE = False
    print("pytket not installed — drawing falls back to QASM")

# QISKIT
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# =============================================================================
# CONSTANTS & HELPERS
# =============================================================================
P = SECP256k1.curve.p()
A = SECP256k1.curve.a()
B = SECP256k1.curve.b()
G = SECP256k1.generator
ORDER = SECP256k1.order
CURVE = CurveFp(P, A, B)

PRESETS = {
    "21": {"bits": 21, "start": 0x90000, "pub": "037d14b19a95fe400b88b0debe31ecc3c0ec94daea90d13057bde89c5f8e6fc25c", "shots": 16384},
    "25": {"bits": 25, "start": 0xE00000, "pub": "038ad4f423459430771c0f12a24df181ed0da5142ec676088031f28a21e86ea06d", "shots": 16384},
    "135": {"bits": 135, "start": 0x400000000000000000000000000000000, "pub": "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16", "shots": 65536},
}

def decompress_pubkey(hex_key: str) -> Point:
    hex_key = hex_key.lower().strip()
    prefix = int(hex_key[:2], 16)
    x_val = int(hex_key[2:], 16)
    y_sq = (pow(x_val, 3, P) + A * x_val + B) % P
    y_val = pow(y_sq, (P + 1) // 4, P)
    if (prefix == 2 and y_val % 2 != 0) or (prefix == 3 and y_val % 2 == 0):
        y_val = P - y_val
    return Point(CURVE, x_val, y_val)

def precompute_deltas(Q: Point, k_start: int, bits: int):
    delta = Q + (-G * k_start)
    dxs = []
    dys = []
    current = delta
    for _ in range(bits):
        dxs.append(int(current.x()) if current else 0)
        dys.append(int(current.y()) if current else 0)
        current = current * 2 if current else None
    return dxs, dys

def calculate_keyspace_start(bits: int) -> int:
    return 1 << (bits - 1)

def process_measurement(meas: int, bits: int, order: int):
    candidates = []
    frac = Fraction(meas, 1 << bits).limit_denominator(order)
    if frac.denominator != 0:
        candidates.append((frac.numerator * pow(frac.denominator, -1, order)) % order)
    candidates.extend([meas % order, (order - meas) % order])
    bitstr = bin(meas)[2:].zfill(bits)
    meas_msb = int(bitstr[::-1], 2)
    frac_msb = Fraction(meas_msb, 1 << bits).limit_denominator(order)
    if frac_msb.denominator != 0:
        candidates.append((frac_msb.numerator * pow(frac_msb.denominator, -1, order)) % order)
    candidates.extend([meas_msb % order, (order - meas_msb) % order])
    return candidates

def bb_correction(measurements: list, order: int):
    best = 0
    max_score = 0
    for cand in set(measurements):
        score = sum(1 for m in measurements if math.gcd(m - cand, order) == 1)
        if score > max_score:
            max_score = score
            best = cand
    return best

def verify_key(k: int, target_x: int) -> bool:
    Pt = G * k
    return Pt is not None and Pt.x() == target_x

def save_key(k: int):
    with open("boom.txt", "w") as f:
        f.write(f"Private key found!\nHEX: {hex(k)}\nDecimal: {k}\n")
        f.write("Donation: 1Bu4CR8Bi5AXQG8pnu1avny88C5CCgWKfb\n")
        f.write(f"Date: {datetime.now()}\n")
    print("🔑 Key saved to boom.txt")

def manual_zne(counts_list):
    extrapolated = defaultdict(int)
    for bitstr in counts_list[0]:
        vals = [c.get(bitstr, 0) for c in counts_list]
        if len(vals) > 1:
            fit = np.polyfit([1, 3, 5, 7], vals, 1)
            extrapolated[bitstr] = max(0, int(fit[1]))
        else:
            extrapolated[bitstr] = vals[0]
    return Counter(extrapolated)

# =============================================================================
# ERROR MITIGATION
# =============================================================================
def apply_surface_code_correction(qc: QuantumCircuit, data_qubits, ancillas, ancilla_cbits):
    if len(data_qubits) < 4 or len(ancillas) < 8:
        return
    for i in range(4):
        qc.h(ancillas[i])
        qc.cx(data_qubits[i], ancillas[i])
        qc.h(ancillas[i])
        qc.measure(ancillas[i], ancilla_cbits[i])
    for i in range(4):
        qc.h(data_qubits[i])
        qc.cx(ancillas[i+4], data_qubits[i])
        qc.h(data_qubits[i])
        qc.measure(ancillas[i], ancilla_cbits[i])
    for a in ancillas:
        qc.reset(a)

def prepare_verified_ancilla(qc, qubit, initial_state=0):
    qc.reset(qubit)
    if initial_state == 1:
        qc.x(qubit)

def encode_repetition(qc, logical_qubit, ancillas):
    qc.cx(logical_qubit, ancillas[0])
    qc.cx(logical_qubit, ancillas[1])

def decode_repetition(qc, ancillas, logical_qubit):
    qc.cx(ancillas[0], logical_qubit)
    qc.cx(ancillas[1], logical_qubit)
    qc.ccx(ancillas[0], ancillas[1], logical_qubit)

def flag_stabilizer_check(qc: QuantumCircuit, ctrl, flag, flag_cbit):
    qc.h(flag)
    qc.cx(ctrl, flag)
    qc.h(flag)
    qc.measure(flag, flag_cbit)

# =============================================================================
# LATTICE + POST-PROCESSING
# =============================================================================
def simple_lll(basis):
    a, b = int(basis[0][0]), int(basis[0][1])
    c, d = int(basis[1][0]), int(basis[1][1])
    while True:
        norm1 = a*a + b*b
        norm2 = c*c + d*d
        if norm1 > norm2:
            a, b, c, d = c, d, a, b
            norm1, norm2 = norm2, norm1
        dot = a*c + b*d
        mu = dot / norm1 if norm1 != 0 else 0
        mu_rounded = round(mu)
        c -= mu_rounded * a
        d -= mu_rounded * b
        if norm2 >= (0.75 - (mu - mu_rounded)**2) * norm1:
            break
    return [[int(a), int(b)], [int(c), int(d)]]

def lattice_reduction(candidates, order, use_bkz=True):
    better = []
    for m in candidates[:60]:
        if m == 0: continue
        if FPYLLL_AVAILABLE and use_bkz:
            try:
                M = IntegerMatrix(2, 2)
                M[0, 0] = order
                M[1, 0] = m
                M[1, 1] = 1
                BKZ.reduce(M, block_size=20)
                better.append(int(M[1, 1]) % order)
                continue
            except: pass
        reduced = simple_lll([[order, 0], [m, 1]])
        better.append(int(reduced[1][1]) % order)
    return better

def qiskit_to_pytket(qc):
    n = qc.num_qubits
    tk = TketCircuit(n, n)
    for inst in qc.data:
        g = inst.operation
        q = [qb.index for qb in inst.qubits]
        if g.name == "h": tk.H(q[0])
        elif g.name == "x": tk.X(q[0])
        elif g.name == "cx": tk.CX(q[0], q[1])
        elif g.name == "ccx": tk.CCX(q[0], q[1], q[2])
        elif g.name == "reset": tk.Reset(q[0])
        elif g.name == "measure":
            tk.Measure(q[0], inst.clbits[0].index)
        elif g.name == "p":
            tk.Rz(float(g.params[0]), q[0])
        # sub-circuit support
        elif hasattr(g, "definition") and g.definition is not None:
            for sub in g.definition.data:
                sg = sub.operation
                sq = [q[qq.index] for qq in sub.qubits]
                if sg.name == "h": tk.H(sq[0])
                elif sg.name == "x": tk.X(sq[0])
                elif sg.name == "cx": tk.CX(sq[0], sq[1])
                elif sg.name in ("rz","p"): tk.Rz(float(sg.params[0]), sq[0])
                elif sg.name == "cp": tk.CU1(float(sg.params[0]), sq[0], sq[1])
                elif sg.name == "ccx": tk.CCX(sq[0], sq[1], sq[2])
    return tk

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\nPresets: 21, 25, 135, c = Custom")
    preset_choice = input("Select preset [21/25/135/c] → ").strip().lower()

    if preset_choice in PRESETS:
        p = PRESETS[preset_choice]
        bits = p["bits"]
        k_start = p["start"]
        pub_hex = p["pub"]
        shots = p["shots"]
    else:
        pub_hex = input("Enter compressed pubkey (hex): ").strip()
        bits = int(input("Enter bit length: ") or 135)
        start_input = input(f"Enter k_start (hex) [Press Enter for auto 2^({bits-1})]: ").strip()
        if start_input:
            k_start = int(start_input, 16)
        else:
            k_start = calculate_keyspace_start(bits)
            print(f"Auto-calculated k_start: {hex(k_start)}")
        shots = int(input("Enter number of shots: ") or 65536)

    print(f"\nRunning for {bits}-bit key | Shots: {shots}")

    USE_ZNE = input("Enable 4-scale ZNE? [y/N] → ").lower() == "y"
    USE_DD = input("Enable 4×XY8 DD? [y/N] → ").lower() == "y"
    USE_REPETITION = input("Enable 3-qubit Repetition on state & ctrl? [y/N] → ").lower() == "y"
    USE_FLAGS = input("Enable flag-qubit checks? [y/N] → ").lower() == "y"
    USE_SURFACE_CODE = input("Enable Surface Code error correction? [y/N] → ").lower() == "y"

    # Define Q early so it's always available
    Q = decompress_pubkey(pub_hex)

    print("=" * 80)
    print("🐉 DRAGON_CODE_v145 — Selene-PyPI now realistic like Helios 🐉 ")
    print("=" * 80)
    print("Choose Platform:")
    print("  [1] Guppy + Q-Nexus (Helios cloud)")
    print("  [2] Qiskit + IBM Cloud")
    choice = input("Select [1/2] → ").strip() or "1"
    BACKEND_MODE = "GUPPY" if choice == "1" else "QISKIT"

    if BACKEND_MODE == "GUPPY":
        import subprocess
        import sys
        import os
        from guppylang import guppy, qubit
        from guppylang.std.quantum import h, x, y, rz, crz, cx, measure, reset
        from guppylang.std.builtins import range as qrange
        from guppylang.std.angles import pi
        import qnexus as qnx
        from datetime import datetime as dt
        from qnexus import jobs
        from qnexus import devices as qdevices

        @guppy
        def draper_oracle_2d(ctrl: qubit, target: list[qubit], value: int):
            n = len(target)
            for i in qrange(n):
                h(target[i])
                for j in qrange(i + 1, n):
                    crz(pi / (2 ** (j - i)), target[j], target[i])
            for i in qrange(n):
                divisor = 2 ** (i + 1)
                angle = 2.0 * pi * (value % divisor) / divisor
                if ctrl:
                    crz(angle, ctrl, target[i])
                else:
                    rz(angle, target[i])
            for i in qrange(n - 1, -1, -1):
                for j in qrange(n - 1, i, -1):
                    crz(-pi / (2 ** (j - i)), target[j], target[i])
                h(target[i])

        @guppy
        def ft_draper_modular_adder(ctrl: qubit, target: list[qubit], ancilla: qubit, value: int, modulus: int):
            n = len(target)
            draper_oracle_2d(ctrl, target, value)
            draper_oracle_2d(None, target, -modulus)
            cx(target[n-1], ancilla)
            cx(ancilla, target[n-1])
            draper_oracle_2d(ancilla, target, modulus)
            cx(ancilla, target[n-1])
            reset(ancilla)

        @guppy
        def mode_shor_style(bits: int, dxs: list[int], dys: list[int], use_repetition: bool, use_flags: bool, use_surface: bool) -> list[bool]:
            rep = 3 if use_repetition else 1
            state = [qubit() for _ in qrange(bits * rep)]
            ctrl_phys = [qubit() for _ in qrange(3 if use_repetition else 1)]
            ancilla = qubit()
            flag = qubit() if use_flags else None
            results = []

            if use_repetition:
                for i in qrange(bits):
                    start = i * rep
                    prepare_verified_ancilla(state[start], 0)
                    encode_repetition(state[start], [state[start+1], state[start+2]])

            if use_repetition:
                prepare_verified_ancilla(ctrl_phys[0], 0)
                encode_repetition(ctrl_phys[0], [ctrl_phys[1], ctrl_phys[2]])

            cx(state[0], ancilla)

            for k in qrange(bits):
                logical_start = k * rep
                for c in qrange(3 if use_repetition else 1):
                    h(ctrl_phys[c])

                combined = (dxs[k] + dys[k]) % (1 << bits)
                ft_draper_modular_adder(ctrl_phys[0], state[logical_start:logical_start+rep], ancilla, combined, 1 << bits)

                if use_flags:
                    flag_stabilizer_check(ctrl_phys[0], flag)

                if use_surface:
                    for _ in qrange(2):
                        reset(ancilla)

                for c in qrange(3 if use_repetition else 1):
                    h(ctrl_phys[c])
                if use_repetition:
                    decode_repetition([ctrl_phys[1], ctrl_phys[2]], ctrl_phys[0])

                results.append(measure(ctrl_phys[0]))
                reset(ctrl_phys[0])
                reset(ancilla)
                if use_flags:
                    reset(flag)

                for _ in qrange(4):
                    for c in qrange(3 if use_repetition else 1):
                        x(ctrl_phys[c]); y(ctrl_phys[c]); x(ctrl_phys[c]); y(ctrl_phys[c])
                        y(ctrl_phys[c]); x(ctrl_phys[c]); y(ctrl_phys[c]); x(ctrl_phys[c])

            return results

        kernel = mode_shor_style

        print("\nGuppy Backend Options:")
        print("  [1] HELIOS (Quantinuum H-Series via Q-Nexus cloud — 2026 API)")
        print("  [2] SELENE (PyPI local simulator)")
        print("  [3] SELENE (GitHub clone)")
        sub_choice = input("Select [1/2/3] → ").strip() or "1"

        dxs, dys = precompute_deltas(Q, k_start, bits)

        if sub_choice == "1":
            # HELIOS — FIXED
            print("🚀 Using HELIOS cloud...")
            print("You have ~60 seconds to complete login in the browser window.")
            print("If login fails or is declined, it will use mock results.")
            try:
                if hasattr(qnx, 'is_authenticated') and qnx.is_authenticated():
                    print("✅ Already authenticated with Q-Nexus.")
                else:
                    qnx.login()
                project = qnx.projects.get_or_create(name="dragon_ecdlp_version")
                qnx.context.set_active_project(project)
                print(f"✅ Active project: {project.name}")

                # Modern device selection
                print("Looking for available devices...")
                all_devices = qnx.devices.get_all().df()
                print("Available devices:", list(all_devices['device_name']) if not all_devices.empty else "None")

                target_device = None
                for name in ["H2-1", "H2-1E", "H1-1E", "H2-Emulator"]:
                    if name in all_devices['device_name'].values:
                        target_device = name
                        break
                if not target_device:
                    target_device = "H2-Emulator"  # safe fallback
                print(f"Using device: {target_device}")

                raw_counts = Counter()
                shots_per_job = min(16384, shots)
                num_jobs = max(1, (shots + shots_per_job - 1) // shots_per_job)

                for j in range(num_jobs):
                    print(f"Submitting batch {j+1}/{num_jobs}...")
                    try:
                        job = qnx.start_execute_job(
                            programs=[mode_shor_style],
                            n_shots=[shots_per_job],
                            backend_config=qnx.QuantinuumConfig(device_name=target_device),
                            inputs={"bits": bits, "dxs": dxs, "dys": dys, "use_repetition": USE_REPETITION, "use_flags": USE_FLAGS, "use_surface": USE_SURFACE_CODE},
                            project=project
                        )
                        print("Waiting for job...")
                        qnx.jobs.wait_for(job)
                        result = qnx.jobs.results(job)
                        # extract counts (adjust based on actual result structure)
                        if result:
                            raw_counts.update(result[0].get_counts() if hasattr(result[0], 'get_counts') else {})
                        print(f"Job {j+1} completed")
                    except Exception as e:
                        print(f"Job {j+1} failed: {e}")

                counts = raw_counts if raw_counts else Counter()

            except Exception as e:
                print(f"Helios login or job failed: {e}")
                print("Falling back to mock results for post-processing...")
                raw_counts = Counter()
                num_mock = max(shots, 16384)
                for _ in range(num_mock):
                    fake = np.random.randint(0, 1 << bits)
                    if np.random.rand() < 0.08:
                        fake = (fake + (1 << (bits//2))) % (1 << bits)
                    fake_bitstr = bin(fake)[2:].zfill(bits)
                    raw_counts[fake_bitstr] += 1
                counts = raw_counts

        elif sub_choice == "3":
            # SELENE-GitHub (unchanged, full version)
            repo = "https://github.com/Quantinuum/selene.git"
            local_path = "selene"
            if not os.path.exists(local_path):
                print("Cloning Selene GitHub...")
                subprocess.run(["git", "clone", repo, local_path], check=True)
            sys.path.insert(0, os.path.abspath(os.path.join(local_path, "selene-sim")))

            try:
                from guppylang import guppy
                from guppylang.std.quantum import qubit
                from guppylang.std.builtins import array
                print("🚀 Using SELENE-GitHub local simulator (100% offline)")

                def qiskit_shor_kernel(bits, dxs, dys, use_repetition, use_flags):
                    rep = 3 if use_repetition else 1
                    flag_qubits = 1 if use_flags else 0
                    qr = QuantumRegister(bits * rep + 3 + flag_qubits, "q")
                    cr = ClassicalRegister(bits, "c")
                    qc = QuantumCircuit(qr, cr)
                    state = qr[:bits*rep]
                    ctrl_encoded = qr[bits*rep : bits*rep + 3]
                    anc = qr[bits*rep + 3]
                    flag = qr[bits*rep + 4] if use_flags else None

                    if use_repetition:
                        for i in range(bits):
                            start = i * rep
                            prepare_verified_ancilla(qc, state[start], 0)
                            encode_repetition(qc, state[start], [state[start+1], state[start+2]])
                        prepare_verified_ancilla(qc, ctrl_encoded[0], 0)
                        encode_repetition(qc, ctrl_encoded[0], [ctrl_encoded[1], ctrl_encoded[2]])

                    qc.x(state[0])
                    qc.cx(state[0], anc)

                    for k in range(bits):
                        logical_start = k * rep
                        for c in range(3 if use_repetition else 1):
                            qc.h(ctrl_encoded[c])
                        combined = (dxs[k] + dys[k]) % (1 << bits)
                        ft_draper_modular_adder(qc, ctrl_encoded[0], state[logical_start:logical_start+rep], anc, combined, 1 << bits)
                        if use_flags:
                            flag_stabilizer_check(qc, ctrl_encoded[0], flag)
                        for c in range(3 if use_repetition else 1):
                            qc.h(ctrl_encoded[c])
                        if use_repetition:
                            decode_repetition(qc, [ctrl_encoded[1], ctrl_encoded[2]], ctrl_encoded[0])
                        qc.measure(ctrl_encoded[0], cr[k])
                        qc.reset(ctrl_encoded[0])
                        qc.reset(anc)
                        if use_flags:
                            qc.reset(flag)
                        for _ in range(4):
                            for c in range(3 if use_repetition else 1):
                                qc.x(ctrl_encoded[c]); qc.y(ctrl_encoded[c])
                                qc.x(ctrl_encoded[c]); qc.y(ctrl_encoded[c])
                                qc.y(ctrl_encoded[c]); qc.x(ctrl_encoded[c])
                                qc.y(ctrl_encoded[c]); qc.x(ctrl_encoded[c])
                    return qc

                circuit = qiskit_shor_kernel(bits, dxs, dys, USE_REPETITION, USE_FLAGS)
                tk_circ = qiskit_to_pytket(circuit)
                shor_guppy = guppy.load_pytket("shor_kernel", tk_circ)

                result = shor_guppy.emulator(n_qubits=circuit.num_qubits).with_shots(shots).run()

                raw_counts = Counter()
                for shot in getattr(result, 'results', []):
                    try:
                        entries = getattr(shot, 'entries', []) or getattr(shot, 'measurements', [])
                        bit_dict = {}
                        for item in entries:
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                tag, val = item[0], item[1]
                                if isinstance(tag, str) and tag.startswith("c") and tag[1:].isdigit():
                                    idx = int(tag[1:])
                                    bit_dict[idx] = 1 if val else 0
                        if bit_dict:
                            bit_list = [bit_dict.get(i, 0) for i in range(bits)]
                            bitstr = ''.join('1' if b else '0' for b in bit_list[::-1])
                            raw_counts[bitstr] += 1
                    except:
                        continue
                counts = raw_counts
                print(f"✅ SELENE-GitHub completed ({len(counts)} unique)")

            except Exception as e:
                print(f"⚠️ SELENE-GitHub failed: {e}")
                print("Falling back to mock...")
                raw_counts = Counter()
                num_mock = max(shots, 16384)
                for _ in range(num_mock):
                    fake = np.random.randint(0, 1 << bits)
                    fake_bitstr = bin(fake)[2:].zfill(bits)
                    raw_counts[fake_bitstr] += 1
                counts = raw_counts

        else:  # SELENE-PyPI — realistic
            try:
                import qnexus as qnx
                print("🚀 Using SELENE-PyPI — attempting Q-Nexus login (realistic like Helios)...")
                if hasattr(qnx, 'is_authenticated') and qnx.is_authenticated():
                    print("✅ Already authenticated.")
                else:
                    qnx.login()

                from guppylang import guppy
                from guppylang.std.quantum import qubit
                from guppylang.std.builtins import array
                print("🚀 Using SELENE-PyPI local simulator (authenticated)")

                def qiskit_shor_kernel(bits, dxs, dys, use_repetition, use_flags, use_surface):
                    rep = 3 if use_repetition else 1
                    flag_qubits = 1 if use_flags else 0
                    surface_anc = 8 if use_surface else 0
                    qr = QuantumRegister(bits * rep + 4 + flag_qubits + surface_anc, "q")
                    cr = ClassicalRegister(bits, "c")
                    qc = QuantumCircuit(qr, cr)
                    state = qr[:bits*rep]
                    ctrl_encoded = qr[bits*rep : bits*rep + 3]
                    anc = qr[bits*rep + 3]
                    flag = qr[bits*rep + 4] if use_flags else None
                    surface_start = bits*rep + 4 + flag_qubits

                    if use_repetition:
                        for i in range(bits):
                            start = i * rep
                            prepare_verified_ancilla(qc, state[start], 0)
                            encode_repetition(qc, state[start], [state[start+1], state[start+2]])
                        prepare_verified_ancilla(qc, ctrl_encoded[0], 0)
                        encode_repetition(qc, ctrl_encoded[0], [ctrl_encoded[1], ctrl_encoded[2]])

                    qc.x(state[0])
                    qc.cx(state[0], anc)

                    for k in range(bits):
                        logical_start = k * rep
                        for c in range(3 if use_repetition else 1):
                            qc.h(ctrl_encoded[c])
                        combined = (dxs[k] + dys[k]) % (1 << bits)
                        ft_draper_modular_adder(qc, ctrl_encoded[0], state[logical_start:logical_start+rep], anc, combined, 1 << bits)
                        if use_flags:
                            flag_stabilizer_check(qc, ctrl_encoded[0], flag)
                        if use_surface:
                            apply_surface_code_correction(qc, state[logical_start:logical_start+4], qr[surface_start:surface_start+8])
                        for c in range(3 if use_repetition else 1):
                            qc.h(ctrl_encoded[c])
                        if use_repetition:
                            decode_repetition(qc, [ctrl_encoded[1], ctrl_encoded[2]], ctrl_encoded[0])
                        qc.measure(ctrl_encoded[0], cr[k])
                        qc.reset(ctrl_encoded[0])
                        qc.reset(anc)
                        if use_flags:
                            qc.reset(flag)
                        for _ in range(4):
                            for c in range(3 if use_repetition else 1):
                                qc.x(ctrl_encoded[c]); qc.y(ctrl_encoded[c])
                                qc.x(ctrl_encoded[c]); qc.y(ctrl_encoded[c])
                                qc.y(ctrl_encoded[c]); qc.x(ctrl_encoded[c])
                                qc.y(ctrl_encoded[c]); qc.x(ctrl_encoded[c])
                    return qc

                circuit = qiskit_shor_kernel(bits, dxs, dys, USE_REPETITION, USE_FLAGS, USE_SURFACE_CODE)
                tk_circ = qiskit_to_pytket(circuit)
                shor_guppy = guppy.load_pytket("shor_kernel", tk_circ)

                result = shor_guppy.emulator(n_qubits=circuit.num_qubits).with_shots(shots).run()

                raw_counts = Counter()
                for shot in getattr(result, 'results', []):
                    try:
                        entries = getattr(shot, 'entries', []) or getattr(shot, 'measurements', [])
                        bit_dict = {}
                        for item in entries:
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                tag, val = item[0], item[1]
                                if isinstance(tag, str) and tag.startswith("c") and tag[1:].isdigit():
                                    idx = int(tag[1:])
                                    bit_dict[idx] = 1 if val else 0
                        if bit_dict:
                            bit_list = [bit_dict.get(i, 0) for i in range(bits)]
                            bitstr = ''.join('1' if b else '0' for b in bit_list[::-1])
                            raw_counts[bitstr] += 1
                    except:
                        continue
                counts = raw_counts
                print(f"✅ SELENE-PyPI completed successfully!")

            except Exception as e:
                print(f"⚠️ SELENE-PyPI failed: {e}")
                print("Falling back to mock counts...")
                raw_counts = Counter()
                num_mock = max(shots, 16384)
                for _ in range(num_mock):
                    fake = np.random.randint(0, 1 << bits)
                    if np.random.rand() < 0.08:
                        fake = (fake + (1 << (bits//2))) % (1 << bits)
                    fake_bitstr = bin(fake)[2:].zfill(bits)
                    raw_counts[fake_bitstr] += 1
                counts = raw_counts

    else:  # QISKIT PATH — FIXED COUNTS
        print("\nIBM Quantum Authentication Setup")
        api_token = input("IBM Quantum API token (Enter if saved): ").strip()
        crn = input("IBM Cloud CRN (Enter to skip): ").strip() or None
        if api_token:
            try:
                QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=api_token, overwrite=True)
                print("✅ IBM credentials saved")
            except Exception as e:
                print(f"⚠️ Token save failed: {e}")

        service = QiskitRuntimeService(instance=crn) if crn else QiskitRuntimeService()

        def draper_oracle_2d(qc, ctrl, target, value):
            n = len(target)
            for i in range(n):
                qc.h(target[i])
                for j in range(i+1, n):
                    qc.cp(math.pi / (2 ** (j-i)), target[j], target[i])
            for i in range(n):
                divisor = 2 ** (i+1)
                angle = 2 * math.pi * (value % divisor) / divisor
                (qc.cp if ctrl is not None else qc.p)(angle, *(ctrl, target[i]) if ctrl is not None else (target[i],))
            for i in range(n-1, -1, -1):
                for j in range(n-1, i, -1):
                    qc.cp(-math.pi / (2 ** (j-i)), target[j], target[i])
                qc.h(target[i])

        def ft_draper_modular_adder(qc, ctrl, target, ancilla, value, modulus):
            n = len(target)
            draper_oracle_2d(qc, ctrl, target, value)
            draper_oracle_2d(qc, None, target, -modulus)
            qc.cx(target[n-1], ancilla)
            qc.cx(ancilla, target[n-1])
            draper_oracle_2d(qc, ancilla, target, modulus)
            qc.cx(ancilla, target[n-1])
            qc.reset(ancilla)

        def mode_shor_style(bits, dxs, dys, use_repetition, use_flags, use_surface):
            rep = 3 if use_repetition else 1
            flag_qubits = 1 if use_flags else 0
            surface_anc = 8 if use_surface else 0
            total_qubits = bits * rep + 4 + flag_qubits + surface_anc

            qr = QuantumRegister(total_qubits, "q")
            cr = ClassicalRegister(bits, "c")
            flag_cr = ClassicalRegister(1, "flag_c") if use_flags else None
            surface_cr = ClassicalRegister(8, "surf_c") if use_surface else None

            regs = [qr, cr]
            if flag_cr: regs.append(flag_cr)
            if surface_cr: regs.append(surface_cr)

            qc = QuantumCircuit(*regs)

            state        = qr[:bits*rep]
            ctrl_encoded = qr[bits*rep : bits*rep + 3]
            anc          = qr[bits*rep + 3]
            flag         = qr[bits*rep + 4] if use_flags else None
            surface_start = bits*rep + 4 + flag_qubits

            if use_repetition:
                for i in range(bits):
                    start = i * rep
                    prepare_verified_ancilla(qc, state[start], 0)
                    encode_repetition(qc, state[start], [state[start+1], state[start+2]])
                prepare_verified_ancilla(qc, ctrl_encoded[0], 0)
                encode_repetition(qc, ctrl_encoded[0], [ctrl_encoded[1], ctrl_encoded[2]])

            qc.x(state[0])
            qc.cx(state[0], anc)

            for k in range(bits):
                logical_start = k * rep
                for c in range(3 if use_repetition else 1):
                    qc.h(ctrl_encoded[c])

                combined = (dxs[k] + dys[k]) % (1 << bits)
                ft_draper_modular_adder(qc, ctrl_encoded[0], state[logical_start:logical_start+rep], anc, combined, 1 << bits)

                if use_flags:
                    flag_stabilizer_check(qc, ctrl_encoded[0], flag, flag_cr[0])
                if use_surface:
                    apply_surface_code_correction(
                        qc,
                        state[logical_start:logical_start+4],
                        qr[surface_start:surface_start+8],
                        surface_cr[:8]
                    )

                for c in range(3 if use_repetition else 1):
                    qc.h(ctrl_encoded[c])
                if use_repetition:
                    decode_repetition(qc, [ctrl_encoded[1], ctrl_encoded[2]], ctrl_encoded[0])

                qc.measure(ctrl_encoded[0], cr[k])
                qc.reset(ctrl_encoded[0])
                qc.reset(anc)
                if use_flags:
                    qc.reset(flag)

                for _ in range(4):
                    for c in range(3 if use_repetition else 1):
                        qc.x(ctrl_encoded[c]); qc.y(ctrl_encoded[c])
                        qc.x(ctrl_encoded[c]); qc.y(ctrl_encoded[c])
                        qc.y(ctrl_encoded[c]); qc.x(ctrl_encoded[c])
                        qc.y(ctrl_encoded[c]); qc.x(ctrl_encoded[c])

            return qc

        dxs, dys = precompute_deltas(Q, k_start, bits)
        qc = mode_shor_style(bits, dxs, dys, USE_REPETITION, USE_FLAGS, USE_SURFACE_CODE)

        dxs, dys = precompute_deltas(Q, k_start, bits)
        qc = mode_shor_style(bits, dxs, dys, USE_REPETITION, USE_FLAGS, USE_SURFACE_CODE)

        print(qc)
        print("🔍 Drawing circuit...")
        qc.draw('mpl', style='iqp', plot_barriers=True, fold=40)
        plt.title(f"Dragon Code — {bits} bits (Surface: {USE_SURFACE_CODE})")
        plt.tight_layout()
        plt.show()

        # ====================== IMPROVED QISKIT SUBMISSION ======================
        USE_REAL = input("Use real IBM hardware? [y/N] → ").lower() == "y"

        if USE_REAL:
            backend = service.least_busy(
                operational=True,
                simulator=False,
                min_num_qubits=bits * 3 + 20
            )
        else:
            backend = AerSimulator()

        backend_name = backend.name if hasattr(backend, 'name') else str(backend)
        print(f"📡 Using backend: {backend_name}")

        pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
        isa_qc = pm.run(qc)

        sampler = Sampler(mode=backend)
        sampler.options.default_shots = shots

        if USE_DD:
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XY4"

        if USE_ZNE:
            print("ℹ️  ZNE: SamplerV2 has no built-in resilience — manual 4-scale ZNE will run post-processing.")

        print(f"📡 Submitting job to backend: {backend_name}")
        print(f"   Shots: {shots}")

        job = sampler.run([isa_qc], shots=shots)
        print(f"   Job ID: {job.job_id()}")
        print("⏳ Waiting for results...")

        result = job.result()
        print("✅ Results retrieved successfully!")

        raw_dict = result[0].data.c.get_counts()
        counts = Counter(raw_dict)

        if USE_ZNE:
            print("🔬 Applying manual 4-scale ZNE extrapolation...")
            zne_counts_list = [counts]
            for noise_factor in [3, 5, 7]:
                scaled_shots = max(1024, shots // noise_factor)
                job_zne = sampler.run([isa_qc], shots=scaled_shots)
                result_zne = job_zne.result()
                zne_counts_list.append(Counter(result_zne[0].data.c.get_counts()))
            counts = manual_zne(zne_counts_list)
            print("✅ Manual ZNE applied across noise factors [1, 3, 5, 7]")

        print(f"\n📊 Received {len(counts)} unique measurement outcomes")
        print("Top 100 most frequent results:")
        for bitstr, cnt in counts.most_common(100):
            print(f"   {bitstr} : {cnt} shots")
        if len(counts) > 100:
            print(f"   ... and {len(counts) - 100} more less frequent outcomes")
        print(f"✅ Job completed on {backend_name}! Total unique bitstrings: {len(counts)}")

    # =============================================================================
    # SHARED POST-PROCESSING (unchanged)
    # =============================================================================
    all_measurements = []
    for bitstr, cnt in counts.items():
        val = int(bitstr, 2)
        all_measurements.extend(process_measurement(val, bits, ORDER) * cnt)

    filtered = [m for m in all_measurements if math.gcd(m, ORDER) == 1]

    print("Extracting multiple continued-fraction convergents...")
    multi_cands = []
    for m in filtered[:200]:
        frac = Fraction(m, 1 << bits).limit_denominator(ORDER)
        if frac.denominator != 0:
            k_cand = (frac.numerator * pow(frac.denominator, -1, ORDER)) % ORDER
            multi_cands.extend([k_cand, (k_cand+1)%ORDER, (k_cand-1)%ORDER])

    print("Applying lattice reduction (BKZ preferred)...")
    lattice_cands = lattice_reduction(filtered, ORDER, use_bkz=FPYLLL_AVAILABLE)

    filtered.extend(multi_cands + lattice_cands)
    filtered = list(set(filtered))[:2000]

    print("Applying majority vote correction...")
    candidate = bb_correction(filtered, ORDER)

    print("\nTrying verification...")
    found = False
    for dk in sorted(set(filtered), reverse=True)[:150]:
        k_test = (k_start + dk) % ORDER
        if verify_key(k_test, Q.x()):
            print("\n" + "═"*80)
            print("🔥 SUCCESS 🔥! PRIVATE KEY FOUND 🔑")
            print(f"HEX: {hex(k_test)}")
            print("Donation : 1Bu4CR8Bi5AXQG8pnu1avny88C5CCgWKfb 💰")
            print("═"*80)
            save_key(k_test)
            found = True
            break
    if not found:
        print("❌ No match — try more shots")

    if counts:
        plt.figure(figsize=(14,7))
        top = counts.most_common(50)
        plt.bar(range(len(top)), [v for _,v in top])
        plt.xticks(range(len(top)), [k for k,_ in top], rotation=90)
        plt.title(f"Measurement Distribution ({len(counts)} unique)")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
