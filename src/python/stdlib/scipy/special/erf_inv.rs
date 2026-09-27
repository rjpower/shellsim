//! Boost's inverse error functions (`boost/math/special_functions/detail/erf_inv.hpp`), which
//! SciPy's `erfinv` calls and which Boost's own incomplete beta and Student's t inverses use as
//! `erfc_inv`. SciPy's `erfcinv` ufunc is the Cephes formula in `erf.rs` instead.
//!
//! Boost evaluates `double` with its approximations for 64-bit types: `x = p (p + 10) (Y + R(p))`
//! for `p <= 1/2`, `x = sqrt(-2 ln q) / (Y + R(q - 1/4))` for `q >= 1/4`, and otherwise
//! `x (Y + R(x - B))` with `x = sqrt(-ln q)` over five ranges of `x`. Each `R` is a ratio of
//! fixed-size polynomials, evaluated with Boost's second-order Horner scheme.
#![allow(clippy::excessive_precision, clippy::unreadable_literal)]

use super::boost;

/// One of Boost's approximations: `Y` plus the ratio of the polynomials `p` and `q`, lowest
/// degree first.
struct ErfInvRange<const P: usize, const Q: usize> {
    y: f64,
    p: [f64; P],
    q: [f64; Q],
}

impl<const P: usize, const Q: usize> ErfInvRange<P, Q> {
    fn ratio(&self, x: f64) -> f64 {
        boost::polynomial(&self.p, x) / boost::polynomial(&self.q, x)
    }

    /// The tail form `x (Y + R(x - start))`, written as Boost rounds it.
    fn tail(&self, x: f64, start: f64) -> f64 {
        self.y * x + self.ratio(x - start) * x
    }
}

const SMALL: ErfInvRange<8, 10> = ErfInvRange {
    y: 0.0891314744949340820313,
    p: [
        -0.000508781949658280665617,
        -0.00836874819741736770379,
        0.0334806625409744615033,
        -0.0126926147662974029034,
        -0.0365637971411762664006,
        0.0219878681111168899165,
        0.00822687874676915743155,
        -0.00538772965071242932965,
    ],
    q: [
        1.0,
        -0.970005043303290640362,
        -1.56574558234175846809,
        1.56221558398423026363,
        0.662328840472002992063,
        -0.71228902341542847553,
        -0.0527396382340099713954,
        0.0795283687341571680018,
        -0.00233393759374190016776,
        0.000886216390456424707504,
    ],
};

const MIDDLE: ErfInvRange<9, 9> = ErfInvRange {
    y: 2.249481201171875,
    p: [
        -0.202433508355938759655,
        0.105264680699391713268,
        8.37050328343119927838,
        17.6447298408374015486,
        -18.8510648058714251895,
        -44.6382324441786960818,
        17.445385985570866523,
        21.1294655448340526258,
        -3.67192254707729348546,
    ],
    q: [
        1.0,
        6.24264124854247537712,
        3.9713437953343869095,
        -28.6608180499800029974,
        -20.1432634680485188801,
        48.5609213108739935468,
        10.8268667355460159008,
        -22.6436933413139721736,
        1.72114765761200282724,
    ],
};

const TAIL_3: ErfInvRange<11, 8> = ErfInvRange {
    y: 0.807220458984375,
    p: [
        -0.131102781679951906451,
        -0.163794047193317060787,
        0.117030156341995252019,
        0.387079738972604337464,
        0.337785538912035898924,
        0.142869534408157156766,
        0.0290157910005329060432,
        0.00214558995388805277169,
        -0.679465575181126350155e-6,
        0.285225331782217055858e-7,
        -0.681149956853776992068e-9,
    ],
    q: [
        1.0,
        3.46625407242567245975,
        5.38168345707006855425,
        4.77846592945843778382,
        2.59301921623620271374,
        0.848854343457902036425,
        0.152264338295331783612,
        0.01105924229346489121,
    ],
};

const TAIL_6: ErfInvRange<9, 7> = ErfInvRange {
    y: 0.93995571136474609375,
    p: [
        -0.0350353787183177984712,
        -0.00222426529213447927281,
        0.0185573306514231072324,
        0.00950804701325919603619,
        0.00187123492819559223345,
        0.000157544617424960554631,
        0.460469890584317994083e-5,
        -0.230404776911882601748e-9,
        0.266339227425782031962e-11,
    ],
    q: [
        1.0,
        1.3653349817554063097,
        0.762059164553623404043,
        0.220091105764131249824,
        0.0341589143670947727934,
        0.00263861676657015992959,
        0.764675292302794483503e-4,
    ],
};

const TAIL_18: ErfInvRange<9, 7> = ErfInvRange {
    y: 0.98362827301025390625,
    p: [
        -0.0167431005076633737133,
        -0.00112951438745580278863,
        0.00105628862152492910091,
        0.000209386317487588078668,
        0.149624783758342370182e-4,
        0.449696789927706453732e-6,
        0.462596163522878599135e-8,
        -0.281128735628831791805e-13,
        0.99055709973310326855e-16,
    ],
    q: [
        1.0,
        0.591429344886417493481,
        0.138151865749083321638,
        0.0160746087093676504695,
        0.000964011807005165528527,
        0.275335474764726041141e-4,
        0.282243172016108031869e-6,
    ],
};

const TAIL_44: ErfInvRange<8, 7> = ErfInvRange {
    y: 0.99714565277099609375,
    p: [
        -0.0024978212791898131227,
        -0.779190719229053954292e-5,
        0.254723037413027451751e-4,
        0.162397777342510920873e-5,
        0.396341011304801168516e-7,
        0.411632831190944208473e-9,
        0.145596286718675035587e-11,
        -0.116765012397184275695e-17,
    ],
    q: [
        1.0,
        0.207123112214422517181,
        0.0169410838120975906478,
        0.000690538265622684595676,
        0.145007359818232637924e-4,
        0.144437756628144157666e-6,
        0.509761276599778486139e-9,
    ],
};

const TAIL_FAR: ErfInvRange<8, 7> = ErfInvRange {
    y: 0.99941349029541015625,
    p: [
        -0.000539042911019078575891,
        -0.28398759004727721098e-6,
        0.899465114892291446442e-6,
        0.229345859265920864296e-7,
        0.225561444863500149219e-9,
        0.947846627503022684216e-12,
        0.135880130108924861008e-14,
        -0.348890393399948882918e-21,
    ],
    q: [
        1.0,
        0.0845746234001899436914,
        0.00282092984726264681981,
        0.468292921940894236786e-4,
        0.399968812193862100054e-6,
        0.161809290887904476097e-8,
        0.231558608310259605225e-11,
    ],
};

/// Boost's `erf_inv_imp`: the `x` with `erf(x) = p` and `erfc(x) = q`, for `p` in `[0, 1]` and
/// `q = 1 - p` passed separately to keep its precision near 1.
fn erf_inv_imp(p: f64, q: f64) -> f64 {
    if p <= 0.5 {
        let g = p * (p + 10.0);
        return g * SMALL.y + g * SMALL.ratio(p);
    }
    if q >= 0.25 {
        let g = (-2.0 * q.ln()).sqrt();
        return g / (MIDDLE.y + MIDDLE.ratio(q - 0.25));
    }
    let x = (-q.ln()).sqrt();
    if x < 3.0 {
        TAIL_3.tail(x, 1.125)
    } else if x < 6.0 {
        TAIL_6.tail(x, 3.0)
    } else if x < 18.0 {
        TAIL_18.tail(x, 6.0)
    } else if x < 44.0 {
        TAIL_44.tail(x, 18.0)
    } else {
        TAIL_FAR.tail(x, 44.0)
    }
}

/// The inverse error function on `[-1, 1]`, as SciPy's `erfinv` computes it with Boost's
/// `erf_inv`. The ends give infinities; arguments outside the domain give NaN.
pub(super) fn erfinv(z: f64) -> f64 {
    if z.is_nan() || !(-1.0..=1.0).contains(&z) {
        return f64::NAN;
    }
    if z == 1.0 {
        return f64::INFINITY;
    }
    if z == -1.0 {
        return f64::NEG_INFINITY;
    }
    if z == 0.0 {
        return 0.0;
    }
    if z < 0.0 {
        -erf_inv_imp(-z, 1.0 + z)
    } else {
        erf_inv_imp(z, 1.0 - z)
    }
}

/// Boost's `erfc_inv` on `[0, 2]`: the `x` with `erfc(x) = z`. Boost reports overflow at the
/// ends; this returns the infinities, and NaN outside the domain.
pub(super) fn erfc_inv(z: f64) -> f64 {
    if z.is_nan() || !(0.0..=2.0).contains(&z) {
        return f64::NAN;
    }
    if z == 0.0 {
        return f64::INFINITY;
    }
    if z == 2.0 {
        return f64::NEG_INFINITY;
    }
    if z > 1.0 {
        let q = 2.0 - z;
        -erf_inv_imp(1.0 - q, q)
    } else {
        erf_inv_imp(1.0 - z, z)
    }
}

#[cfg(test)]
mod tests {
    use super::super::erf::{erf, erfc};
    use super::*;

    #[test]
    fn inverses_round_trip_through_each_range() {
        // erfc keeps full precision in the upper tail and erf near 0.
        for x in [0.6, 1.0, 1.5, 2.5, 5.0, 12.0, 26.0] {
            let z = erfc(x);
            assert!(
                (erfc_inv(z) - x).abs() <= 1e-14 * x,
                "erfc_inv({z}) for {x}"
            );
        }
        for x in [1e-300, 1e-8, 0.3, 0.6, 1.0, 2.5] {
            let y = erf(x);
            assert!((erfinv(y) - x).abs() <= 1e-13 * x, "erfinv({y}) for {x}");
            assert_eq!(erfinv(-y), -erfinv(y));
            // 1 - y rounds, so near 0 only an absolute error bound holds.
            assert!((erfc_inv(1.0 - y) - x).abs() <= 1e-13 * x.max(1e-2), "{x}");
        }
    }

    #[test]
    fn domain_ends_and_errors() {
        assert_eq!(erfinv(1.0), f64::INFINITY);
        assert_eq!(erfinv(-1.0), f64::NEG_INFINITY);
        assert_eq!(erfinv(0.0).to_bits(), 0.0f64.to_bits());
        assert!(erfinv(1.5).is_nan() && erfinv(f64::NAN).is_nan());
        assert_eq!(erfc_inv(0.0), f64::INFINITY);
        assert_eq!(erfc_inv(2.0), f64::NEG_INFINITY);
        assert!(erfc_inv(-0.5).is_nan());
    }
}
