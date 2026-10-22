//! Pure graph geometry for the gpui app: sample series → polygon vertices.
//!
//! Coordinates are `(f32, f32)` in canvas space, y down, origin top-left.
//! The paint closure converts them to `Point<Pixels>`.

use super::data::MemSample;

/// Stacked memory layers, bottom-first in VRAM order: graphics (video),
/// compute (LLM), other (unaccounted). Each layer is a closed polygon's
/// vertex list.
pub(crate) struct StackedGeometry {
    pub graphics: Vec<(f32, f32)>,
    pub compute: Vec<(f32, f32)>,
    pub other: Vec<(f32, f32)>,
    /// Scale maximum in bytes (VRAM size, or the largest stacked sum).
    /// Read by tests; the label row computes the same value independently.
    #[allow(dead_code)]
    pub y_max: u64,
}

/// Build the stacked memory polygons across `width`×`height`.
///
/// Samples spread evenly over the full width (a single sample stretches to a
/// full-width band). `None` when there is nothing to draw yet or the scale
/// would divide by zero.
pub(crate) fn stacked_geometry(
    samples: &[MemSample],
    width: f32,
    height: f32,
) -> Option<StackedGeometry> {
    if samples.is_empty() || width <= 0.0 || height <= 0.0 {
        return None;
    }
    let y_max = samples
        .iter()
        .map(|s| s.mem_total.max(s.compute + s.graphics + s.other))
        .max()?;
    if y_max == 0 {
        return None;
    }
    let scale = height / y_max as f32;

    // Draw slots: (sample index, x). A single sample becomes two slots at
    // x=0 and x=width so the band still spans the full width.
    let len = samples.len();
    let slots: Vec<(usize, f32)> = if len == 1 {
        vec![(0, 0.0), (0, width)]
    } else {
        (0..len)
            .map(|i| (i, width * i as f32 / (len - 1) as f32))
            .collect()
    };

    let mut graphics = Vec::with_capacity(2 * slots.len());
    let mut compute = Vec::with_capacity(2 * slots.len());
    let mut other = Vec::with_capacity(2 * slots.len());
    // Top edge left→right, bottom edge right→left (closed polygon).
    for &(idx, x) in &slots {
        let s = &samples[idx];
        let g_h = s.graphics as f32 * scale;
        let c_h = s.compute as f32 * scale;
        // Clamp `other` so rounding can never push the stack past the top.
        let o_h = (s.other as f32 * scale).min(height - g_h - c_h).max(0.0);
        graphics.push((x, height - g_h));
        compute.push((x, height - g_h - c_h));
        other.push((x, height - g_h - c_h - o_h));
    }
    for &(idx, x) in slots.iter().rev() {
        let s = &samples[idx];
        let g_h = s.graphics as f32 * scale;
        let c_h = s.compute as f32 * scale;
        graphics.push((x, height));
        compute.push((x, height - g_h));
        other.push((x, height - g_h - c_h));
    }

    Some(StackedGeometry {
        graphics,
        compute,
        other,
        y_max,
    })
}

/// Single-series area polygon (utilization, power). `None` when empty or the
/// scale maximum is not positive.
pub(crate) fn area_geometry(
    values: &[f32],
    v_max: f32,
    width: f32,
    height: f32,
) -> Option<Vec<(f32, f32)>> {
    if values.is_empty() || width <= 0.0 || height <= 0.0 || v_max <= 0.0 {
        return None;
    }
    let len = values.len();
    let slots: Vec<(usize, f32)> = if len == 1 {
        vec![(0, 0.0), (0, width)]
    } else {
        (0..len)
            .map(|i| (i, width * i as f32 / (len - 1) as f32))
            .collect()
    };
    let mut polygon = Vec::with_capacity(2 * slots.len());
    for &(idx, x) in &slots {
        let y = height - (values[idx] / v_max * height).clamp(0.0, height);
        polygon.push((x, y));
    }
    for &(_, x) in slots.iter().rev() {
        polygon.push((x, height));
    }
    Some(polygon)
}

/// Gridline spacing on the memory graph: 500 MiB steps.
pub(crate) const GRID_STEP: u64 = 500 * 1_048_576;
const GIB: u64 = 1_048_576 * 1024;

/// Gridline values below `y_max` (the top edge itself carries no line).
pub(crate) fn grid_ticks(y_max: u64) -> Vec<u64> {
    (1..)
        .map(|k| k * GRID_STEP)
        .take_while(|&v| v < y_max)
        .collect()
}

/// Labeled gridline values: GiB multiples up to and including `y_max`,
/// thinned so at most ~8 labels sit on the axis without overlapping.
pub(crate) fn grid_labels(y_max: u64) -> Vec<u64> {
    if y_max < GIB {
        return Vec::new();
    }
    let step = (y_max / GIB).div_ceil(8).max(1) * GIB;
    (1..)
        .map(|k| k * step)
        .take_while(|&v| v <= y_max)
        .collect()
}

/// Compact axis label for a gridline value: GiB multiples are bare numbers
/// (the unit lives in the y_max label at the top), sub-GiB steps carry theirs.
pub(crate) fn grid_label(bytes: u64) -> String {
    if bytes.is_multiple_of(GIB) {
        format!("{}", bytes / GIB)
    } else {
        format!("{}M", bytes / 1_048_576)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(mem_total: u64, compute: u64, graphics: u64, other: u64) -> MemSample {
        MemSample {
            mem_total,
            compute,
            graphics,
            other,
            util: 0,
            power: 0,
        }
    }

    #[test]
    fn test_stacked_geometry_empty_and_zero() {
        assert!(stacked_geometry(&[], 100.0, 100.0).is_none());
        let zero = [sample(0, 0, 0, 0)];
        assert!(stacked_geometry(&zero, 100.0, 100.0).is_none());
        let s = [sample(1000, 100, 200, 300)];
        assert!(stacked_geometry(&s, 0.0, 100.0).is_none());
    }

    #[test]
    fn test_stacked_geometry_layers_ordered() {
        // 1000 total: graphics 200 (bottom), compute 300 (middle), other 100.
        let samples = [sample(1000, 300, 200, 100)];
        let geo = stacked_geometry(&samples, 100.0, 50.0).unwrap();
        assert_eq!(geo.y_max, 1000);
        // Single sample stretches to two slots at x=0 and x=100.
        // graphics height = 200/1000*50 = 10 → top edge at y=40.
        assert_eq!(geo.graphics[0], (0.0, 40.0));
        assert_eq!(geo.graphics[1], (100.0, 40.0));
        // compute sits on top of graphics: 300/1000*50 = 15 → top at y=25.
        assert_eq!(geo.compute[0].1, 25.0);
        // other: 100/1000*50 = 5 → top at y=20.
        assert_eq!(geo.other[0].1, 20.0);
        // Bottom edges (reverse slots): graphics on the baseline, each layer
        // on the top of the one below; the walk ends back at x=0.
        assert_eq!(geo.graphics[2], (100.0, 50.0));
        assert_eq!(geo.compute[2].1, 40.0);
        assert_eq!(geo.other[2].1, 25.0);
        assert_eq!(*geo.graphics.last().unwrap(), (0.0, 50.0));
    }

    #[test]
    fn test_stacked_geometry_two_samples_spread_x() {
        let samples = [sample(1000, 0, 0, 0), sample(1000, 0, 0, 0)];
        let geo = stacked_geometry(&samples, 90.0, 10.0).unwrap();
        assert_eq!(geo.graphics[0].0, 0.0);
        assert_eq!(geo.graphics[1].0, 90.0);
    }

    #[test]
    fn test_stacked_geometry_other_fits_under_total() {
        // mem_total dominates the scale; other (500) fits between the
        // compute top and the graph top.
        let samples = [sample(1000, 10, 10, 500)];
        let geo = stacked_geometry(&samples, 100.0, 50.0).unwrap();
        // scale = 50/1000; graphics+compute take 0.5 each, other 25.
        assert_eq!(geo.other[0].1, 24.0);
    }

    #[test]
    fn test_area_geometry_basic() {
        assert!(area_geometry(&[], 100.0, 100.0, 50.0).is_none());
        assert!(area_geometry(&[50.0], 0.0, 100.0, 50.0).is_none());
        let poly = area_geometry(&[0.0, 100.0], 100.0, 100.0, 50.0).unwrap();
        // First point: value 0 → at baseline; second: value 100 → at top.
        assert_eq!(poly[0], (0.0, 50.0));
        assert_eq!(poly[1], (100.0, 0.0));
        assert_eq!(poly[2], (100.0, 50.0));
        assert_eq!(poly[3], (0.0, 50.0));
    }

    #[test]
    fn test_grid_ticks_step_500mib() {
        let gib = 1_048_576u64 * 1024;
        // 6 GiB / 500 MiB = 12.3 → twelve full steps fit below the top edge.
        let ticks = grid_ticks(6 * gib);
        assert_eq!(ticks.len(), 12);
        assert_eq!(ticks[0], GRID_STEP);
        assert_eq!(*ticks.last().unwrap(), 12 * GRID_STEP);
        assert!(grid_ticks(0).is_empty());
    }

    #[test]
    fn test_grid_labels_gib_multiples_thinned() {
        let gib = 1_048_576u64 * 1024;
        // Up to and including y_max — the top of the bar is labeled too.
        assert_eq!(
            grid_labels(6 * gib),
            vec![gib, 2 * gib, 3 * gib, 4 * gib, 5 * gib, 6 * gib]
        );
        // Below one GiB there is no room for a labeled gridline.
        assert!(grid_labels(gib / 2).is_empty());
        // 24 GiB thins to 3-GiB steps so the labels stay readable.
        assert_eq!(
            grid_labels(24 * gib),
            (1..=8).map(|k| 3 * k * gib).collect::<Vec<_>>()
        );
        // 9-13 GiB must thin too (12 GiB → 2-GiB steps, 6 labels) or the
        // axis overflows with overlapping text.
        assert_eq!(
            grid_labels(12 * gib),
            (1..=6).map(|k| 2 * k * gib).collect::<Vec<_>>()
        );
        assert_eq!(
            grid_labels(13 * gib),
            (1..=6).map(|k| 2 * k * gib).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_grid_label_format() {
        let gib = 1_048_576u64 * 1024;
        // GiB multiples render as bare numbers; the unit lives in the y_max label.
        assert_eq!(grid_label(gib), "1");
        assert_eq!(grid_label(GRID_STEP), "500M");
        assert_eq!(grid_label(gib / 2), "512M");
    }
}
