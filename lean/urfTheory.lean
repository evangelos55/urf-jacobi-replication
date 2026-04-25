import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.LinearAlgebra.Matrix.PosDef

open Matrix
open scoped Matrix

/-!
# URF-3 : Géométrie de la Suture et Stress Résiduel
Ce module formalise la stabilité du prix spot autour du prix géométrique
sur le manifold Sym+(100) via la courbure de Ricci.
CONJECTURE : En régime de courbure positive (Compression), l'erreur σ
est rappelée vers l'attracteur géodésique (Suture).
-/

/-- Dimension de l'univers (100 séries temporelles pour le SP500). -/
def n : ℕ := 100

/-- Espace des matrices de covariance Sym+(100).
Représente la métrique de Fisher-Rao sur le manifold des matrices PD. -/
def SymPositiveDefinite := { M : Matrix (Fin n) (Fin n) ℝ // M.IsSymm ∧ M.PosDef }

-- Déclaration des variables au niveau global
variable (V : ℝ → ℝ) (S : ℝ) (V0 : ℝ)

/-- La Vitesse de Suture λ est proportionnelle à la courbure de Ricci. -/
noncomputable def suture_velocity (S : ℝ) : ℝ := S / (n : ℝ)

/-- RÉGIME 1 : Compression (Force Géométrique). -/
def is_compression_regime (S : ℝ) : Prop := S > 0

/-- RÉGIME 2 : Suture (Convergence). -/
def is_suture_regime (V : ℝ → ℝ) (S : ℝ) : Prop :=
  is_compression_regime S ∧ Differentiable ℝ V ∧ ∀ t, deriv V t = -S * V t

/-- RÉGIME 3 : Rupture (Singularité de Minsky). -/
def is_rupture_regime (V : ℝ → ℝ) (S : ℝ) : Prop :=
  S < 0 ∧ Differentiable ℝ V ∧ ∀ t, deriv V t = -S * V t

/-- THÉORÈME 10.1 (Suture) : Convergence vers l'attracteur géodésique. -/
theorem suture_convergence (h_suture : is_suture_regime V S) (hV0 : V 0 = V0) :
    ∀ ε > 0, ∃ T, ∀ t > T, V t < ε := by
  rcases h_suture with ⟨hS_pos, hV_diff, h_deriv⟩
  intro ε hε
  -- 1. Identification de la trajectoire
  have h_sol : ∀ t, V t = V0 * Real.exp (-S * t) := by
    have hg_deriv : ∀ t, deriv (fun s => V s * Real.exp (S * s)) t = 0 := by
      intro t
      have h1 : HasDerivAt V (-S * V t) t := by
        have hd := (hV_diff t).hasDerivAt
        rw [h_deriv t] at hd
        exact hd
      have h2 : HasDerivAt (fun s => Real.exp (S * s)) (S * Real.exp (S * t)) t := by
        have inner : HasDerivAt (fun s => S * s) S t := by
          have := (hasDerivAt_id t).const_mul S
          simp only [id, mul_one] at this
          exact this
        have := (Real.hasDerivAt_exp (S * t)).comp t inner
        convert this using 1
        ring
      have h3 : HasDerivAt (fun s => V s * Real.exp (S * s)) 0 t := by
        have := h1.mul h2
        convert this using 1
        ring
      exact h3.deriv
    have hg_diff : Differentiable ℝ (fun s => V s * Real.exp (S * s)) :=
      hV_diff.mul (Real.differentiable_exp.comp
        ((differentiable_const S).mul differentiable_id))
    intro t
    have hconst : V t * Real.exp (S * t) = V0 := by
      have h := is_const_of_deriv_eq_zero hg_diff hg_deriv t 0
      simp only [mul_zero, Real.exp_zero, mul_one] at h
      rw [hV0] at h
      exact h
    rw [show -S * t = -(S * t) from by ring, Real.exp_neg, ← div_eq_mul_inv]
    exact (eq_div_iff (Real.exp_ne_zero _)).mpr hconst
  -- 2. Analyse de la limite
  by_cases hV0_pos : V0 > 0
  · use (Real.log (V0 / ε)) / S
    intro t ht
    rw [h_sol]
    have h1 : -S * t < Real.log (ε / V0) := by
      have hlog : Real.log (V0 / ε) = -Real.log (ε / V0) := by rw [← Real.log_inv, inv_div]
      have hmul : Real.log (V0 / ε) < t * S := (div_lt_iff₀ hS_pos).mp ht
      linarith [hlog ▸ hmul]
    have h2 : Real.exp (-S * t) < ε / V0 :=
      calc Real.exp (-S * t)
         < Real.exp (Real.log (ε / V0)) := Real.exp_lt_exp.mpr h1
        _ = ε / V0                       := Real.exp_log (div_pos hε hV0_pos)
    calc V0 * Real.exp (-S * t)
        < V0 * (ε / V0) := mul_lt_mul_of_pos_left h2 hV0_pos
      _ = ε             := by field_simp
  · use 0
    intro t ht
    rw [h_sol]
    have h_exp_pos : 0 < Real.exp (-S * t) := Real.exp_pos _
    have h_nonpos : V0 * Real.exp (-S * t) ≤ 0 :=
      mul_nonpos_of_nonpos_of_nonneg (not_lt.mp hV0_pos) h_exp_pos.le
    exact h_nonpos.trans_lt hε

/-- THÉORÈME 10.2 (Rupture) : Divergence du Stress Résiduel. -/
theorem rupture_divergence (h_rupture : is_rupture_regime V S) (hV0 : V 0 = V0) (hV0_pos : V0 > 0) :
    ∀ M, ∃ T, ∀ t > T, V t > M := by
  rcases h_rupture with ⟨hS_neg, hV_diff, h_deriv⟩
  have hS_pos : 0 < -S := neg_pos.mpr hS_neg
  -- 1. Même ODE : V(t) = V0·exp(-S·t)
  have h_sol : ∀ t, V t = V0 * Real.exp (-S * t) := by
    have hg_deriv : ∀ t, deriv (fun s => V s * Real.exp (S * s)) t = 0 := by
      intro t
      have h1 : HasDerivAt V (-S * V t) t := by
        have hd := (hV_diff t).hasDerivAt
        rw [h_deriv t] at hd
        exact hd
      have h2 : HasDerivAt (fun s => Real.exp (S * s)) (S * Real.exp (S * t)) t := by
        have inner : HasDerivAt (fun s => S * s) S t := by
          have := (hasDerivAt_id t).const_mul S
          simp only [id, mul_one] at this
          exact this
        have := (Real.hasDerivAt_exp (S * t)).comp t inner
        convert this using 1
        ring
      have h3 : HasDerivAt (fun s => V s * Real.exp (S * s)) 0 t := by
        have := h1.mul h2
        convert this using 1
        ring
      exact h3.deriv
    have hg_diff : Differentiable ℝ (fun s => V s * Real.exp (S * s)) :=
      hV_diff.mul (Real.differentiable_exp.comp
        ((differentiable_const S).mul differentiable_id))
    intro t
    have hconst : V t * Real.exp (S * t) = V0 := by
      have h := is_const_of_deriv_eq_zero hg_diff hg_deriv t 0
      simp only [mul_zero, Real.exp_zero, mul_one] at h
      rw [hV0] at h
      exact h
    rw [show -S * t = -(S * t) from by ring, Real.exp_neg, ← div_eq_mul_inv]
    exact (eq_div_iff (Real.exp_ne_zero _)).mpr hconst
  -- 2. Divergence
  intro M
  by_cases hM : M ≤ 0
  · use 0
    intro t _ht
    rw [h_sol]
    exact lt_of_le_of_lt hM (mul_pos hV0_pos (Real.exp_pos _))
  · push_neg at hM
    use Real.log (M / V0) / (-S)
    intro t ht
    rw [h_sol]
    have h1 : Real.log (M / V0) < -S * t := by
      have hmul : Real.log (M / V0) < t * (-S) := (div_lt_iff₀ hS_pos).mp ht
      linarith
    have h2 : M / V0 < Real.exp (-S * t) :=
      calc M / V0
          = Real.exp (Real.log (M / V0)) := (Real.exp_log (div_pos hM hV0_pos)).symm
        _ < Real.exp (-S * t)            := Real.exp_lt_exp.mpr h1
    calc M = V0 * (M / V0) := by field_simp
      _ < V0 * Real.exp (-S * t) := mul_lt_mul_of_pos_left h2 hV0_pos

--------------------------------------------------------------------------------
-- SECTION DIAGNOSTIC
--------------------------------------------------------------------------------
#check suture_convergence
#print axioms suture_convergence
#check suture_velocity
