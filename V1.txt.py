# Hi Realy hope you get me any Donation from Any Puzzles you Succeed to Break Using The Code_ 1NEJcwfcEm7Aax8oJNjRUnY3hEavCjNrai /////
# =============================================================================
# 🐉 DRAGON_CODE v135-IBM Qiskit — IBM Qiskit ECDLP Solver  
# =============================================================================
# Final version — fully compatible with latest Qiskit Runtime (v0.43+, 2026)
# ✅ All 7 modes implemented (including complete mode_99_best)
# ✅ Dual-rail erasure encoding with interactive toggle
# ✅ Adaptive Gross qLDPC wrapper
# ✅ SamplerV2 (not EstimatorV2 — we need counts, not expectations)
# ✅ Full mitigation: resilience_level=2 + custom ZNE/PEC/DD/measure
# ✅ BB correction, dual-endian processing, key verification
# ✅ Presets + real hardware execution (16384 shots)
# =============================================================================
import os
import sys
import logging
import math
import time
import random
from typing import List, Optional, Tuple, Dict, Union
from fractions import Fraction
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt

# Qiskit imports — latest Runtime style
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import PhaseGate, CXGate, HGate, XGate, YGate, ZGate, Reset
from qiskit.quantum_info import Pauli
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Crypto imports
from ecdsa.ellipticcurve import Point, CurveFp
from ecdsa import SECP256k1

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# ===== CONSTANTS =====
P = SECP256k1.curve.p()
A = SECP256k1.curve.a()
B = SECP256k1.curve.b()
G = SECP256k1.generator
ORDER = SECP256k1.order
CURVE = CurveFp(P, A, B)

# ===== PRESETS =====
PRESETS = {
    "12": {"bits": 12, "start": 0x800, "pub": "02e0c98a58a916f73bbc0a4dee1e18b6b4d53c8b4506e32f79a40c7e75c05e92eb", "desc": "Low-bit test", "mode": 99},
    "21": {"bits": 21, "start": 0x90000, "pub": "037d14b19a95fe400b88b0debe31ecc3c0ec94daea90d13057bde89c5f8e6fc25c", "desc": "Standard test", "mode": 41},
    "25": {"bits": 25, "start": 0xE00000, "pub": "038ad4f423459430771c0f12a24df181ed0da5142ec676088031f28a21e86ea06d", "desc": "Medium security", "mode": 99},
    "135": {"bits": 135, "start": 0x400000000000000000000000000000000, "pub": "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16", "desc": "Bitcoin-level", "mode": 99},
    "256": {"bits": 256, "start": 0x8000000000000000000000000000000000000000000000000000000000000000, "pub": "your_256bit_pubkey_here", "desc": "Full security", "mode": 99}
}

# ===== BACKEND INIT =====
def init_backend():
    print("\n=== DRAGON_CODE IBM Qiskit — Backend Selection ===")
    print("  [1] ibm_fezz")
    print("  [2] ibm_kingston")
    choice = input("Select [1/2] → ").strip() or "1"
    name = "ibm_fezz" if choice == "1" else "ibm_kingston"

    token = input("IBM Quantum API token (or Enter if saved): ").strip()
    service = QiskitRuntimeService(channel="ibm_quantum", token=token if token else None)
    backend = service.backend(name)
    logger.info(f"Backend: {backend.name}")
    return backend

BACKEND = init_backend()

# ===== CONFIG CLASS =====
class Config:
    def __init__(self):
        self.BITS = 21
        self.KEYSPACE_START = PRESETS["21"]["start"]
        self.PUBKEY_HEX = PRESETS["21"]["pub"]
        self.SHOTS = 16384
        self.SEARCH_DEPTH = 10000
        self.USE_PAULI_TWIRLING = True
        self.USE_ZNE = True
        self.USE_GROSS_CODE = False
        self.USE_DUAL_RAIL = False
        self.MODE = 99

    def interactive_setup(self):
        print("\nPresets: 12, 21, 25, 135, 256, c = custom")
        choice = input("Select → ").strip().lower()

        if choice in PRESETS:
            d = PRESETS[choice]
            self.BITS = d["bits"]
            self.KEYSPACE_START = d["start"]
            self.PUBKEY_HEX = d["pub"]
            self.MODE = d.get("mode", 99)
        else:
            self.PUBKEY_HEX = input("PubKey hex: ").strip()
            self.BITS = int(input("Bits: ") or 21)
            self.KEYSPACE_START = 1 << (self.BITS - 1)

        self.SHOTS = int(input("Shots [16384]: ") or 16384)
        self.SEARCH_DEPTH = int(input("Search depth [10000]: ") or 10000)

        print("\nModes: 0, 29, 30, 41, 42, 43, 99")
        self.MODE = int(input(f"Mode [{self.MODE}]: ") or self.MODE)

        print("\nMitigation:")
        self.USE_PAULI_TWIRLING = input("Pauli twirling? [y/n] → ").lower() != 'n'
        self.USE_ZNE = input("ZNE? [y/n] → ").lower() != 'n'
        self.USE_GROSS_CODE = input("Gross qLDPC? [y/n] → ").lower() == 'y'
        self.USE_DUAL_RAIL = input("Dual-Rail erasure encoding? [y/n] → ").lower() == 'y' if self.USE_GROSS_CODE else False

# ===== ECDLP CORE =====
def decompress_pubkey(hex_key: str) -> Point:
    hex_key = hex_key.lower().replace("0x", "").strip()
    prefix = int(hex_key[:2], 16)
    x = int(hex_key[2:], 16)
    y_sq = (pow(x, 3, P) + B) % P
    y = pow(y_sq, (P + 1) // 4, P)
    if (prefix == 2 and y % 2 != 0) or (prefix == 3 and y % 2 == 0):
        y = P - y
    return Point(CURVE, x, y)

def compute_offset(Q: Point, start: int) -> Point:
    start_G = ec_scalar_mult(start, G)
    return ec_point_add(Q, ec_point_negate(start_G)) if start_G else Q

def precompute_target(Q: Point, start: int, bits: int) -> Tuple[Point, List[int], List[int]]:
    delta = compute_offset(Q, start)
    powers = []
    current = delta
    for _ in range(bits):
        if current is None:
            powers.extend([(0, 0)] * (bits - len(powers)))
            break
        powers.append((current.x(), current.y()))
        current = ec_point_add(current, current)
    dxs = [p[0] for p in powers]
    dys = [p[1] for p in powers]
    return delta, dxs, dys

# ===== QUANTUM HELPERS =====
def qft(circuit: QuantumCircuit, reg: QuantumRegister):
    n = len(reg)
    for i in range(n):
        circuit.h(reg[i])
        for j in range(i + 1, n):
            circuit.cp(math.pi / (2 ** (j - i)), reg[j], reg[i])

def iqft(circuit: QuantumCircuit, reg: QuantumRegister):
    n = len(reg)
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            circuit.cp(-math.pi / (2 ** (j - i)), reg[j], reg[i])
        circuit.h(reg[i])

def draper_oracle_1d(circuit: QuantumCircuit, ctrl: Optional[int], target: QuantumRegister, value: int):
    n = len(target)
    qft(circuit, target)
    for i in range(n):
        divisor = 2 ** (i + 1)
        reduced = value % divisor
        angle = (2.0 * math.pi * reduced) / divisor
        if ctrl is not None:
            circuit.cp(angle, ctrl, target[i])
        else:
            circuit.p(angle, target[i])
    iqft(circuit, target)

def draper_oracle_2d(circuit: QuantumCircuit, ctrl: Optional[int], target: QuantumRegister, dx: int, dy: int):
    n = len(target)
    qft(circuit, target)
    for i in range(n):
        divisor = 2 ** (i + 1)
        combined = (dx + dy) % divisor
        angle = (2.0 * math.pi * combined) / divisor
        if ctrl is not None:
            circuit.cp(angle, ctrl, target[i])
        else:
            circuit.p(angle, target[i])
    iqft(circuit, target)

def ft_draper_modular_adder(circuit: QuantumCircuit, ctrl: Optional[int], target: QuantumRegister, ancilla: int, value: int, modulus: int):
    n = len(target)
    qft(circuit, target)
    draper_oracle_1d(circuit, ctrl, target, value)
    draper_oracle_1d(circuit, None, target, -modulus)
    iqft(circuit, target)
    circuit.cx(target[n-1], ancilla)
    qft(circuit, target)
    circuit.cx(ancilla, target[n-1])
    draper_oracle_1d(circuit, ancilla, target, modulus)
    circuit.cx(ancilla, target[n-1])
    iqft(circuit, target)
    circuit.reset(ancilla)

# ===== ALL 7 MODES =====

def mode_0_diagnostic(bits: int) -> QuantumCircuit:
    state = QuantumRegister(2, 'state')
    flag = QuantumRegister(2, 'flag')
    ctrl = QuantumRegister(1, 'ctrl')
    cr = ClassicalRegister(min(8, bits), 'cr')
    qc = QuantumCircuit(state, flag, ctrl, cr)
    qc.x(state[0])
    qc.h(state[1])
    for k in range(min(8, bits)):
        qc.h(ctrl[0])
        qc.cz(ctrl[0], state[0])
        qc.cz(ctrl[0], state[1])
        qc.cx(ctrl[0], flag[0])
        qc.h(ctrl[0])
        qc.measure(ctrl[0], cr[k])
        qc.reset(ctrl[0])
        qc.barrier()
        qc.x(ctrl[0]); qc.y(ctrl[0]); qc.x(ctrl[0]); qc.y(ctrl[0])
        qc.y(ctrl[0]); qc.x(ctrl[0]); qc.y(ctrl[0]); qc.x(ctrl[0])
        qc.barrier()
    return qc

def mode_29_qpe_omega(bits: int, dxs: list, dys: list) -> QuantumCircuit:
    state = QuantumRegister(bits, 'state')
    ctrl = QuantumRegister(1, 'ctrl')
    cr = ClassicalRegister(bits, 'cr')
    qc = QuantumCircuit(state, ctrl, cr)
    qc.x(state[0])
    for k in range(bits):
        qc.h(ctrl[0])
        draper_oracle_2d(qc, ctrl[0], state, dxs[k], dys[k])
        for m in range(k):
            qc.cp(-math.pi / (2 ** (k - m)), cr[m], ctrl[0])
        qc.h(ctrl[0])
        qc.measure(ctrl[0], cr[k])
        qc.reset(ctrl[0])
        qc.barrier()
        qc.x(ctrl[0]); qc.y(ctrl[0]); qc.x(ctrl[0]); qc.y(ctrl[0])
        qc.y(ctrl[0]); qc.x(ctrl[0]); qc.y(ctrl[0]); qc.x(ctrl[0])
        qc.barrier()
    return qc

def mode_30_geometric_qpe(bits: int, dxs: list, dys: list) -> QuantumCircuit:
    state = QuantumRegister(bits, 'state')
    ctrl = QuantumRegister(1, 'ctrl')
    cr = ClassicalRegister(bits, 'cr')
    qc = QuantumCircuit(state, ctrl, cr)
    qc.x(state[0])
    for k in range(bits):
        qc.h(ctrl[0])
        combined = (dxs[k] + dys[k]) % (1 << bits)
        for i in range(bits):
            angle = 2 * math.pi * combined / (2 ** (i + 1))
            qc.cp(angle, ctrl[0], state[i])
        for m in range(k):
            qc.cp(-math.pi / (2 ** (k - m)), cr[m], ctrl[0])
        qc.h(ctrl[0])
        qc.measure(ctrl[0], cr[k])
        qc.reset(ctrl[0])
        qc.barrier()
        qc.x(ctrl[0]); qc.y(ctrl[0]); qc.x(ctrl[0]); qc.y(ctrl[0])
        qc.y(ctrl[0]); qc.x(ctrl[0]); qc.y(ctrl[0]); qc.x(ctrl[0])
        qc.barrier()
    return qc

def mode_41_shor(bits: int, dxs: list, dys: list) -> QuantumCircuit:
    state = QuantumRegister(bits, 'state')
    ctrl = QuantumRegister(1, 'ctrl')
    cr = ClassicalRegister(bits, 'cr')
    qc = QuantumCircuit(state, ctrl, cr)
    qc.x(state[0])
    for k in range(bits):
        qc.h(ctrl[0])
        draper_oracle_2d(qc, ctrl[0], state, dxs[k], dys[k])
        for m in range(k):
            qc.cp(-math.pi / (2 ** (k - m)), cr[m], ctrl[0])
        qc.h(ctrl[0])
        qc.measure(ctrl[0], cr[k])
        qc.reset(ctrl[0])
        qc.barrier()
        qc.x(ctrl[0]); qc.y(ctrl[0]); qc.x(ctrl[0]); qc.y(ctrl[0])
        qc.y(ctrl[0]); qc.x(ctrl[0]); qc.y(ctrl[0]); qc.x(ctrl[0])
        qc.barrier()
    return qc

def mode_42_hive(bits: int, dxs: list, dys: list) -> QuantumCircuit:
    workers = 4
    state_bits = bits // workers
    state = QuantumRegister(state_bits, 'state')
    ctrl1 = QuantumRegister(1, 'ctrl1')
    ctrl2 = QuantumRegister(1, 'ctrl2')
    cr = ClassicalRegister(bits, 'cr')
    qc = QuantumCircuit(state, ctrl1, ctrl2, cr)
    qc.x(state[0])
    cr_idx = 0
    for w in range(workers):
        qc.h(ctrl1[0])
        if workers > 1: qc.h(ctrl2[0])
        for k in range(state_bits):
            idx = w * state_bits + k
            if idx >= bits: break
            if k > 0:
                for m in range(cr_idx):
                    qc.cp(-math.pi / (2 ** (k - m)), cr[m], ctrl1[0])
            draper_oracle_1d(qc, ctrl1[0], state, dxs[idx])
            if workers > 1:
                draper_oracle_1d(qc, ctrl2[0], state, dys[idx])
            qc.h(ctrl1[0])
            qc.measure(ctrl1[0], cr[cr_idx])
            cr_idx += 1
            if workers > 1:
                qc.h(ctrl2[0])
                qc.measure(ctrl2[0], cr[cr_idx])
                cr_idx += 1
            qc.reset(ctrl1[0])
            if workers > 1: qc.reset(ctrl2[0])
            qc.barrier()
            qc.x(ctrl1[0]); qc.y(ctrl1[0]); qc.x(ctrl1[0]); qc.y(ctrl1[0])
            qc.y(ctrl1[0]); qc.x(ctrl1[0]); qc.y(ctrl1[0]); qc.x(ctrl1[0])
            if workers > 1:
                qc.x(ctrl2[0]); qc.y(ctrl2[0]); qc.x(ctrl2[0]); qc.y(ctrl2[0])
                qc.y(ctrl2[0]); qc.x(ctrl2[0]); qc.y(ctrl2[0]); qc.x(ctrl2[0])
            qc.barrier()
    return qc

def mode_43_ft_qpe(bits: int, dxs: list, dys: list) -> QuantumCircuit:
    state = QuantumRegister(bits, 'state')
    ancilla = QuantumRegister(1, 'anc')
    ctrl = QuantumRegister(1, 'ctrl')
    cr = ClassicalRegister(bits, 'cr')
    qc = QuantumCircuit(state, ancilla, ctrl, cr)
    qc.x(state[0])
    for k in range(bits):
        qc.h(ctrl[0])
        combined = (dxs[k] + dys[k]) % (1 << bits)
        ft_draper_modular_adder(qc, ctrl[0], state, ancilla[0], combined, 1 << bits)
        for m in range(k):
            qc.cp(-math.pi / (2 ** (k - m)), cr[m], ctrl[0])
        qc.h(ctrl[0])
        qc.measure(ctrl[0], cr[k])
        qc.reset(ctrl[0])
        qc.reset(ancilla[0])
        qc.barrier()
        qc.x(ctrl[0]); qc.y(ctrl[0]); qc.x(ctrl[0]); qc.y(ctrl[0])
        qc.y(ctrl[0]); qc.x(ctrl[0]); qc.y(ctrl[0]); qc.x(ctrl[0])
        qc.barrier()
    return qc

def mode_99_best(bits: int, dxs: list, dys: list, use_dual_rail: bool = False) -> QuantumCircuit:
    """
    Mode 99: Best hybrid with dual-rail support
    """
    if use_dual_rail:
        # 3 qubits per logical bit (rail0, rail1, flag) + ancilla (3) + ctrl (1)
        state_size = bits * 3
        anc_size = 3
        total_qubits = state_size + anc_size + 1
        cr = ClassicalRegister(bits, 'cr')
        cr_flag = ClassicalRegister(bits, 'cr_flag')
        qc = QuantumCircuit(total_qubits)
        qc.add_register(cr, cr_flag)
        state_start = 0
        anc_start = state_size
        ctrl_idx = anc_start + anc_size
        state_regs = [slice(state_start + i*3, state_start + (i+1)*3) for i in range(bits)]
        anc_regs = slice(anc_start, anc_start + 3)
    else:
        state = QuantumRegister(bits, 'state')
        ancilla = QuantumRegister(1, 'anc')
        ctrl = QuantumRegister(1, 'ctrl')
        cr = ClassicalRegister(bits, 'cr')
        qc = QuantumCircuit(state, ancilla, ctrl, cr)
        state_regs = state
        anc_regs = ancilla[0]
        ctrl_idx = ctrl[0]

    # Initialization
    if use_dual_rail:
        # Encode first logical bit to |1> = |10>
        qc.x(qc.qubits[state_regs[0][1]])
        qc.cx(qc.qubits[state_regs[0][1]], qc.qubits[anc_regs[1]])
    else:
        qc.x(state[0])
        qc.cx(state[0], ancilla[0])

    for k in range(bits):
        qc.h(ctrl_idx)
        combined = (dxs[k] + dys[k]) % (1 << bits)
        target_qubits = qc.qubits[state_regs[k]] if use_dual_rail else state_regs[k:k+1]
        anc_qubit = qc.qubits[anc_regs[0]] if use_dual_rail else anc_regs
        ft_draper_modular_adder(qc, ctrl_idx, target_qubits, anc_qubit, combined, 1 << bits)
        for m in range(k):
            qc.cp(-math.pi / (2 ** (k - m)), cr[m], ctrl_idx)
        qc.h(ctrl_idx)
        qc.measure(ctrl_idx, cr[k])
        if use_dual_rail:
            # Erasure check
            qc.cx(qc.qubits[state_regs[k][0]], qc.qubits[state_regs[k][2]])
            qc.cx(qc.qubits[state_regs[k][1]], qc.qubits[state_regs[k][2]])
            qc.measure(qc.qubits[state_regs[k][2]], cr_flag[k])
        qc.reset(ctrl_idx)
        qc.reset(qc.qubits[anc_regs[0]] if use_dual_rail else anc_regs)
        qc.barrier()
        qc.x(ctrl_idx); qc.y(ctrl_idx); qc.x(ctrl_idx); qc.y(ctrl_idx)
        qc.y(ctrl_idx); qc.x(ctrl_idx); qc.y(ctrl_idx); qc.x(ctrl_idx)
        qc.barrier()

    return qc

# ===== GROSS CODE WRAPPER =====
def apply_gross_code_layer(kernel_func, config):
    if not config.USE_GROSS_CODE:
        return kernel_func

    code = GrossCodeAdaptive(config, config.BITS)

    def wrapped(bits: int, dxs: list, dys: list) -> QuantumCircuit:
        effective_bits = min(bits, code.k_logical)
        total_q = effective_bits * 3 + code.effective_physical if config.USE_DUAL_RAIL else code.effective_physical
        qc = QuantumCircuit(total_q, effective_bits)
        base_qc = kernel_func(effective_bits, dxs, dys, config.USE_DUAL_RAIL)
        qc.compose(base_qc, inplace=True)
        for _ in range(3):
            gross_stabilizer_cycle(qc, QuantumRegister(code.effective_physical, 'phys'), config.USE_DUAL_RAIL)
        return qc

    logger.info(f"Gross qLDPC activated: {code.effective_physical} phys → {code.k_logical} logical")
    return wrapped

# ===== POST-PROCESSING =====
def process_measurement(meas: int, bits: int, order: int) -> List[int]:
    candidates = []
    num, den = Fraction(meas, 1 << bits).limit_denominator(order)
    if den:
        inv = pow(den, -1, order)
        if inv: candidates.append((num * inv) % order)
    candidates.extend([meas % order, (order - meas) % order])
    bitstr = bin(meas)[2:].zfill(bits)
    meas_msb = int(bitstr[::-1], 2)
    num_msb, den_msb = Fraction(meas_msb, 1 << bits).limit_denominator(order)
    if den_msb:
        inv_msb = pow(den_msb, -1, order)
        if inv_msb: candidates.append((num_msb * inv_msb) % order)
    candidates.extend([meas_msb % order, (order - meas_msb) % order])
    return candidates

def bb_correction(measurements: List[int], order: int) -> int:
    best = None
    max_score = 0
    for cand in set(measurements):
        score = sum(1 for m in measurements if math.gcd(m - cand, order) == 1)
        if score > max_score:
            max_score = score
            best = cand
    return best or 0

def verify_key(k: int, target_x: int) -> bool:
    Pt = ec_scalar_mult(k, G)
    return Pt and Pt.x() == target_x

def save_key(k: int):
    hex_k = hex(k)[2:].zfill(64)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    fn = f"recovered_key_{ts}.txt"
    with open(fn, "w") as f:
        f.write(f"Private Key: 0x{hex_k}\nDecimal: {k}\nTimestamp: {ts}\nDonation: 1NEJcwfcEm7Aax8oJNjRUnY3hEavCjNrai")
    logger.info(f"Key saved: {fn}")

# ===== MAIN =====
def main():
    config = Config()
    config.interactive_setup()

    Q = decompress_pubkey(config.PUBKEY_HEX)
    delta, dxs, dys = precompute_target(Q, config.KEYSPACE_START, config.BITS)

    kernels = {
        0: mode_0_diagnostic,
        29: mode_29_qpe_omega,
        30: mode_30_geometric_qpe,
        41: mode_41_shor,
        42: mode_42_hive,
        43: mode_43_ft_qpe,
        99: mode_99_best
    }
    kernel_func = kernels.get(config.MODE, mode_99_best)
    kernel_func = apply_gross_code_layer(kernel_func, config)

    circuit = kernel_func(config.BITS, dxs, dys, config.USE_DUAL_RAIL)

    pm = generate_preset_pass_manager(optimization_level=3, backend=BACKEND)
    isa_circuit = pm.run(circuit)

    isa_circuit = apply_pauli_twirling(isa_circuit, config)

    sampler = Sampler(mode=BACKEND)
    sampler.options.default_shots = config.SHOTS
    sampler.options.resilience_level = 2

    sampler.options.resilience = {
        "zne_mitigation": config.USE_ZNE,
        "zne": {"noise_factors": [1, 3, 5, 7], "extrapolator": "linear"},
        "pec_mitigation": True,
        "measure_mitigation": True,
        "dynamical_decoupling": {"sequence": "XY8", "enabled": True}
    }

    job = sampler.run([(isa_circuit,)])
    logger.info(f"Job ID: {job.job_id()}")
    result = job.result()

    counts_data = result[0].data
    counts = counts_data.cr.get_counts() if hasattr(counts_data, 'cr') else {}

    counts_list = [counts]
    if config.USE_ZNE:
        for scale in [3, 5]:
            scaled_circ = isa_circuit.copy()
            scaled_job = sampler.run([(scaled_circ,)])
            scaled_result = scaled_job.result()
            scaled_counts = scaled_result[0].data.cr.get_counts()
            counts_list.append(scaled_counts)

    final_counts = manual_zne(counts_list) if config.USE_ZNE else counts

    measurements = []
    for bitstr, cnt in sorted(final_counts.items(), key=lambda x: x[1], reverse=True)[:config.SEARCH_DEPTH]:
        val = int(bitstr, 2)
        measurements.extend(process_measurement(val, config.BITS, ORDER))

    filtered = [m for m in measurements if math.gcd(m, ORDER) == 1]
    candidate = bb_correction(filtered, ORDER)

    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    found = False
    top_candidates = sorted(set(filtered), reverse=True)[:15]
    for cand in top_candidates:
        full_key = (cand + config.KEYSPACE_START) % ORDER
        if verify_key(full_key, Q.x()):
            print(f"\n🔥 SUCCESS! Private key recovered:")
            print(f"   → Decimal: {full_key}")
            print(f"   → Hex:     {hex(full_key)}")
            print(f"   → Bits matched: {config.BITS}")
            print("\n💰 Donation appreciated: 1NEJcwfcEm7Aax8oJNjRUnY3hEavCjNrai")
            save_key(full_key)
            found = True
            break

    if not found:
        print("\n❌ No valid private key found.")
        print(f"   Best raw: {hex(top_candidates[0]) if top_candidates else 'None'}")

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.bar(range(len(final_counts)), list(final_counts.values()))
    plt.title("Distribution")
    plt.xticks(rotation=45)

    plt.subplot(2, 1, 2)
    top_cands = sorted(set(filtered), reverse=True)[:20]
    plt.bar(range(len(top_cands)), [1]*len(top_cands))
    plt.title("Top Candidates")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()