import Mathlib

-- variable (n m a π_ML : ℝ)

-- theorem derivation_1
--   (n m a π_ML : ℝ)
--   (h : m * (1 / (a * π_ML + (1-a) * (1-π_ML))) * (a - (1 - a)) = -(n - m) * (1 / ((1-a) * π_ML + a * (1-π_ML))) * ((1 - a) - a)) :
--   π_ML = (a - 1 + (m/n)) / (2 * a - 1) :=
-- by
-- have h' : m * (1 / (a * π_ML + (1-a) * (1-π_ML))) * (a - (1 - a))
--       = (n - m) * (1 / ((1-a) * π_ML + a * (1-π_ML))) * (a - (1 - a)) :=
--   by
--     rw [eq_neg_iff_eq_neg] at h
--     rw [neg_mul_eq_mul_neg, ←neg_eq_iff_eq_neg] at h
--     -- (1-a) - a = -(a - (1-a)), so we get sign flip:
--     rw [neg_mul_eq_mul_neg, ←neg_eq_iff_eq_neg, mul_assoc, mul_assoc, ←neg_one_mul] at h
--     -- rhs: -(n-m) * (...) * (-(a-(1-a))) = (n-m) * (...) * (a-(1-a))
--     simp only [neg_neg] at h
--     exact h,
-- -- End Generation Here
-- ```

lemma small_test
  (n m a : ℝ)
  (h : m * (a - (1 - a)) = -(n - m) * ((1 - a) - a)) :
  m * (a - (1 - a)) = (n - m) * (a - (1 - a)) :=
by
  -- Note: (1 - a) - a = -(a - (1 - a)), so -(n-m) * ((1-a) - a) = (n-m) * (a - (1-a))
  linarith
