import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import UrfTheory

/-!
# URF-WIZARD: Two-Price Theory and Strike Certification

This module formalises the competitive advantage of OMAV:
Geometric Price (G) calculation and strike optimization via the Options Wizard.
-/

open Real

variable (P G S : ℝ) (hP : P > 0) (hG : G > 0) (hS : S > 0)

/-- 1. TWO-PRICE DEFINITION

`residual_stress P G` measures the log-relative deviation between the market price `P`
and the geometric price `G`. -/
noncomputable def residual_stress (P G : ℝ) : ℝ := log (P / G)

/-- 2. OPTIONS WIZARD FORMULA

`geodesic_strike Spot σ` computes the certified strike price `K` based on
the current spot price `Spot` and the residual stress `σ`.

Formula: `K = Spot * exp (-σ)`. -/
noncomputable def geodesic_strike (Spot σ : ℝ) : ℝ := Spot * exp (-σ)

/-- THEOREM 11.1 (Target Alignment)

If the market price equals the geometric price, the residual stress is zero. -/
theorem geodesic_strike_alignment (G : ℝ) (hG : G > 0) :
    residual_stress G G = 0 := by
  simp only [residual_stress]
  rw [div_self hG.ne.symm, log_one]

/-- THEOREM 11.2 (Wizard Consistency)

If we feed the residual stress into the geodesic strike formula,
we recover the geometric price `G`. -/
theorem wizard_recalibration (S G : ℝ) (hS : S > 0) (hG : G > 0) :
    geodesic_strike S (residual_stress S G) = G := by
  simp only [geodesic_strike, residual_stress]
  rw [log_div hS.ne.symm hG.ne.symm, neg_sub, exp_sub, exp_log hG, exp_log hS]
  field_simp [hS.ne.symm]

/-- THEOREM 11.3 (Zero-Stress Arbitrage)

If we use the Wizard to recalibrate the strike, the resulting residual stress
with respect to `G` is zero. -/
theorem wizard_zero_stress (S G : ℝ) (hS : S > 0) (hG : G > 0) :
    let σ := residual_stress S G
    let K := geodesic_strike S σ
    residual_stress K G = 0 := by
  intro σ K
  simp only [residual_stress, K, geodesic_strike, σ]
  rw [log_div hS.ne.symm hG.ne.symm, neg_sub, exp_sub, exp_log hG, exp_log hS]
  field_simp [hS.ne.symm, hG.ne.symm]
  rw [log_one]

/-- 3. GAP TO EQUILIBRIUM

`gap_to_equilibrium P G` measures the relative gap between `P` and `G`. -/
noncomputable def gap_to_equilibrium (P G : ℝ) : ℝ := (G - P) / P

/-- 4. LYAPUNOV POTENTIAL (ENERGY)

`manifold_energy V G` is the Lyapunov-like energy associated with the
log-relative deviation between `V` and `G`:

`E = 1/2 * (log (V / G))^2`. -/
noncomputable def manifold_energy (V G : ℝ) : ℝ :=
  (1 / 2) * (log (V / G))^2

/-- THEOREM 12.1 (Suture Stability - Lyapunov)

We consider a "suture" dynamics on a positive state variable `V(t)`:

*Relative suture speed:* `deriv V t = -S * V t`.

Under this dynamics, the manifold energy

`E(t) = 1/2 * (log (V t / G))^2`

has derivative

`E'(t) = (log (V t / G)) * (-S)`,

which is negative whenever `log (V t / G)` and `S` have the same sign.
This certifies Lyapunov-type decay of the energy along the suture flow. -/
theorem suture_energy_decay
    (V : ℝ → ℝ) (S G : ℝ) (hS : S > 0) (hG : G > 0)
    (h_diffV : ∀ t, DifferentiableAt ℝ V t)
    (h_suture : ∀ t, deriv V t = -S * V t)
    (t : ℝ) (hVt : V t > 0) :
    deriv (fun t => (1 / 2) * (log (V t / G))^2) t = (log (V t / G)) * (-S) := by

  --------------------------------------------------------------------
  -- A. Basic non-vanishing facts for the logarithm argument V(t)/G
  --------------------------------------------------------------------
  have hVne : V t ≠ 0 := ne_of_gt hVt
  have hGne : G ≠ 0 := ne_of_gt hG
  have hVG_ne : V t / G ≠ 0 := div_ne_zero hVne hGne

  --------------------------------------------------------------------
  -- B. Differentiability of the inner map and of the log block
  --------------------------------------------------------------------
  -- Inner map: t ↦ V t / G
  have h_inner_diff : DifferentiableAt ℝ (fun t => V t / G) t :=
    (h_diffV t).div_const G

  -- Logarithmic block: t ↦ log (V t / G)
  have h_log_diff : DifferentiableAt ℝ (fun t => log (V t / G)) t := by
    apply DifferentiableAt.log
    · exact h_inner_diff
    · exact hVG_ne

  --------------------------------------------------------------------
  -- C. Derivative of the energy: E(t) = 1/2 * (log (V t / G))^2
  --------------------------------------------------------------------
  -- Step C1: Pull out the constant 1/2 using `deriv_const_mul`
  have h_const :
      deriv (fun t => (1 / 2) * (log (V t / G))^2) t =
        (1 / 2) * deriv (fun t => (log (V t / G))^2) t :=
    deriv_const_mul (1 / 2 : ℝ) (h_log_diff.pow 2)

  -- Step C2: Differentiate the square using `deriv_fun_pow`
  have h_pow :
      deriv (fun t => (log (V t / G))^2) t =
        (2 : ℝ) * log (V t / G) * deriv (fun t => log (V t / G)) t := by
    -- Apply the general power rule for functions
    have h := deriv_fun_pow h_log_diff 2
    -- h : deriv (fun x => log (V x / G) ^ 2) t
    --        = (2 : ℝ) * log (V t / G) ^ (2 - 1)
    --            * deriv (fun t => log (V t / G)) t

    -- Rewrite 2 - 1 to 1
    have h1 : (2 - 1 : ℕ) = 1 := by decide
    -- Replace the exponent
    rw [h1] at h

    -- Rewrite a^1 = a
    have hpow1 : log (V t / G) ^ 1 = log (V t / G) := by
      simp
    rw [hpow1] at h

    -- Now h has exactly the desired shape
    exact h

  -- Step C3: Combine C1 and C2 to get the derivative of E(t)
  have h_energy_deriv :
      deriv (fun t => (1 / 2) * (log (V t / G))^2) t =
        (1 / 2) * (2 * log (V t / G) *
          deriv (fun t => log (V t / G)) t) := by
    -- Start from the constant factor formula
    have h := h_const
    -- Replace the derivative of the square using h_pow
    rw [h_pow] at h
    -- The result is exactly the desired expression
    exact h

  --------------------------------------------------------------------
  -- D. Derivative of the log block via the chain rule
  --------------------------------------------------------------------
  -- Step D1: HasDerivAt for log at the point V t / G
  -- Step D1: HasDerivAt for log at the point V t / G
  have h_log_has :
      HasDerivAt (fun x => log x) ((V t / G)⁻¹) (V t / G) :=
    hasDerivAt_log hVG_ne

  -- Step D2: HasDerivAt for the inner map t ↦ V t / G
  have h_inner_has := h_inner_diff.hasDerivAt

  -- Step D3: Chain rule for t ↦ log (V t / G)
  have h_log_comp_has := h_log_has.comp t h_inner_has

  -- Step D4: Convert HasDerivAt to a raw statement about `deriv`
  have h_log_comp_raw :
      deriv ((fun x => log x) ∘ fun t => V t / G) t =
        (V t / G)⁻¹ * deriv (fun t => V t / G) t :=
    h_log_comp_has.deriv

  -- Step D5: Explicit derivative of t ↦ log (V t / G)
  have h_log_deriv_explicit :
      deriv (fun t => log (V t / G)) t =
        (V t / G)⁻¹ * deriv (fun t => V t / G) t := by
    -- h_log_comp_raw gives the derivative of the composition
    -- ((fun x => log x) ∘ fun t => V t / G)
    -- but this is definitionally equal to fun t => log (V t / G)
    simpa using h_log_comp_raw


  --------------------------------------------------------------------
  -- E. Substitute the log derivative and inject the suture dynamics
  --------------------------------------------------------------------
  -- Step E1: put the derivative of log(V t / G) in explicit form
  have h_energy_deriv' :
    deriv (fun t => 1 / 2 * log (V t / G) ^ 2) t =
      1 / 2 * (2 * log (V t / G) *
        (1 / (V t / G) * (deriv V t / G))) := by
      -- on part de l'expression déjà obtenue
      have h := h_energy_deriv
      -- dérivée de l'intérieur : t ↦ V t / G
      have h_inner_deriv :
          deriv (fun t => V t / G) t = deriv V t / G := by
        simpa using h_inner_has.deriv
      -- dérivée de log(V t / G) via la règle de chaîne (D5)
      have h_log_comp_explicit :
          deriv (fun t => log (V t / G)) t =
            1 / (V t / G) * (deriv V t / G) := by
        have h0 := h_log_deriv_explicit
        rw [h_inner_deriv] at h0
        simpa [one_div] using h0
      -- substitution dans la dérivée d'énergie
      rw [h_log_comp_explicit] at h
      exact h


  -- Step E2: Use the suture dynamics: V' = -S * V
  have h_sut := h_suture t

  -- Step E3: Final algebraic simplification to reach log(V t / G) * (-S)
  have h_final :
      (1 / 2) * (2 * log (V t / G) *
        ((1 / (V t / G)) * (deriv V t / G))) =
        log (V t / G) * (-S) := by
    -- Replace deriv V t by -S * V t
    rw [h_sut]
    -- Clear denominators and normalize
    field_simp [div_eq_mul_inv, hGne, hVne, hVG_ne]
    -- Use ring_nf instead of ring
  ring_nf

   -- Step E4: Combine the analytic and algebraic parts in Lean's normalized form
  have h_goal :
      deriv (fun t => log (V t * G⁻¹) ^ 2 * (1 / 2)) t =
        -(log (G⁻¹ * V t) * S) := by
    -- Start from the “human” equality we proved
    have h := h_energy_deriv'.trans h_final
    -- Rewrite both sides into Lean’s normalized form
    simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using h

  exact h_goal
