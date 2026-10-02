//! Heap tests drive allocation and collection directly, with explicit root sets, so they can
//! force young and full collections and observe moves, reclamation and accounting.

use super::*;
use crate::resources::{Limits, Resources};

fn unlimited() -> Resources {
    Resources::new(Limits::unlimited())
}

/// A list holding `items`, built the way the VM does: through a builder inside the allocation.
fn list<'s>(
    heap: &mut Heap,
    roots: &mut dyn Roots,
    resources: &mut Resources,
    items: &[Value<'s>],
) -> Value<'s> {
    heap.alloc_with(roots, resources, |builder| {
        Object::List(builder.refs(items.iter().copied()))
    })
    .unwrap()
}

fn string<'s>(
    heap: &mut Heap,
    roots: &mut dyn Roots,
    resources: &mut Resources,
    text: &str,
) -> Value<'s> {
    heap.alloc(Object::String(text.into()), roots, resources)
        .unwrap()
}

fn text(heap: &Heap, value: Value<'_>) -> String {
    match heap.get(value).unwrap() {
        Object::String(text) => text.to_string(),
        other => panic!("expected a string, found {other:?}"),
    }
}

#[test]
fn text_and_byte_payloads_are_charged_per_byte() {
    assert_eq!(
        modeled_size(&Object::Bytes(vec![0; 1000])),
        Ok(OBJECT_HEADER + 1000)
    );
    assert_eq!(
        modeled_size(&Object::ByteArray(vec![0; 10])),
        Ok(OBJECT_HEADER + 10)
    );
    assert_eq!(
        modeled_size(&Object::String("é".repeat(5).into())),
        Ok(OBJECT_HEADER + 10)
    );
    let mut heap = Heap::default();
    let mut resources = unlimited();
    let items = [Value::None; 10];
    let value = list(&mut heap, &mut (), &mut resources, &items);
    assert_eq!(
        heap.object_bytes(value).unwrap(),
        OBJECT_HEADER + 10 * MODELED_VALUE_BYTES
    );
}

#[test]
fn handles_follow_objects_through_a_young_collection() {
    let mut heap = Heap::default();
    let mut resources = unlimited();
    let kept = string(&mut heap, &mut (), &mut resources, "kept");
    let dropped = string(&mut heap, &mut (), &mut resources, "dropped");
    let dropped_slot = heap.store(dropped);
    heap.truncate_handles(1);
    assert_eq!(heap.live_objects(), 2);

    heap.collect_young(&mut (), None, &mut resources).unwrap();
    assert_eq!(heap.live_objects(), 1);
    assert_eq!(text(&heap, kept), "kept");
    assert_eq!(heap.stats().promoted_objects, 1);
    // A reference that was not a root points into the discarded young epoch.
    assert_eq!(
        heap.get(heap.handle(&dropped_slot)).unwrap_err(),
        "stale reference to a moved young object"
    );
}

#[test]
fn reported_roots_are_rewritten_and_keep_their_targets() {
    let mut heap = Heap::default();
    let mut resources = unlimited();
    let mut roots: Vec<Ref> = Vec::new();
    let value = string(&mut heap, &mut roots, &mut resources, "rooted");
    roots.push(heap.store(value));
    heap.truncate_handles(0);

    heap.collect_young(&mut roots, None, &mut resources)
        .unwrap();
    let value = heap.handle(&roots[0]);
    assert_eq!(text(&heap, value), "rooted");
    assert_eq!(heap.live_objects(), 1);
}

#[test]
fn references_stored_into_old_objects_are_remembered() {
    let mut heap = Heap::default();
    let mut resources = unlimited();
    let container = list(&mut heap, &mut (), &mut resources, &[]);
    heap.collect_young(&mut (), None, &mut resources).unwrap();
    // `container` is old now; store a young string into it without any other root.
    let young = string(&mut heap, &mut (), &mut resources, "young");
    heap.modify(container, |builder, object| {
        let Object::List(items) = object else {
            unreachable!()
        };
        items.push(builder.store(young));
    })
    .unwrap();
    heap.truncate_handles(1);

    heap.collect_young(&mut (), None, &mut resources).unwrap();
    let Object::List(items) = heap.get(container).unwrap() else {
        unreachable!()
    };
    let element = heap.handle(&items[0]);
    assert_eq!(text(&heap, element), "young");
}

#[test]
fn full_collection_reclaims_unreachable_cycles_and_releases_memory() {
    let mut heap = Heap::default();
    let mut resources = unlimited();
    let first = list(&mut heap, &mut (), &mut resources, &[]);
    let second = list(&mut heap, &mut (), &mut resources, &[first]);
    heap.reserve_object_growth(first, MODELED_VALUE_BYTES, &mut (), &mut resources)
        .unwrap();
    heap.modify(first, |builder, object| {
        let Object::List(items) = object else {
            unreachable!()
        };
        items.push(builder.store(second));
    })
    .unwrap();

    // Both are reachable from the handle stack.
    heap.collect_full(&mut (), None, &mut resources).unwrap();
    assert_eq!(heap.live_objects(), 2);
    heap.truncate_handles(0);
    let released = heap.collect_full(&mut (), None, &mut resources).unwrap();
    assert!(released > 0);
    assert_eq!(heap.live_objects(), 0);
    assert_eq!(heap.modeled_bytes(), 0);
    assert_eq!(resources.outcome(0, 0, 0).usage.memory_current, 0);
}

#[test]
fn the_object_under_construction_survives_the_collection_it_triggers() {
    let mut heap = Heap::default();
    let limit = 8 * 1024 * 1024;
    let mut resources = Resources::new(Limits {
        memory: limit,
        ..Limits::unlimited()
    });
    let element = string(&mut heap, &mut (), &mut resources, "element");
    // Fill the young space with empty byte objects until less than one of them fits in the
    // budget. A one-element list costs more than an empty object, so allocating the container is
    // what triggers the collection.
    let before = heap.young_bytes;
    heap.alloc(Object::Bytes(Vec::new()), &mut (), &mut resources)
        .unwrap();
    let garbage_cost = heap.young_bytes - before;
    while heap.young_bytes + garbage_cost <= gc::young_budget(limit) {
        heap.alloc(Object::Bytes(Vec::new()), &mut (), &mut resources)
            .unwrap();
        heap.truncate_handles(1);
    }
    let minor_before = heap.stats().minor_collections;
    let container = list(&mut heap, &mut (), &mut resources, &[element]);
    assert_eq!(
        heap.stats().minor_collections,
        minor_before + 1,
        "allocation collected the young space"
    );
    assert!(heap.young.len() <= 2);
    let Object::List(items) = heap.get(container).unwrap() else {
        unreachable!()
    };
    let element = heap.handle(&items[0]);
    assert_eq!(text(&heap, element), "element");
}

#[test]
fn allocation_collects_before_reporting_out_of_memory() {
    let mut heap = Heap::default();
    let mut resources = Resources::new(Limits {
        memory: 16 * 1024,
        ..Limits::unlimited()
    });
    for _ in 0..100 {
        heap.alloc(Object::Bytes(vec![0; 1024]), &mut (), &mut resources)
            .unwrap();
        heap.truncate_handles(0);
    }
    assert!(heap.stats().major_collections > 0);
    // Live data beyond the limit is still an error.
    let mut roots: Vec<Ref> = Vec::new();
    let error = loop {
        match heap.alloc(Object::Bytes(vec![0; 1024]), &mut roots, &mut resources) {
            Ok(value) => roots.push(heap.store(value)),
            Err(error) => break error,
        }
        heap.truncate_handles(0);
    };
    assert_eq!(error, "memory limit exceeded");
    assert!(roots.len() >= 8 && roots.len() < 16);
}

#[test]
fn collection_charges_cpu_for_every_object() {
    let mut heap = Heap::default();
    let mut allocation = unlimited();
    for _ in 0..3 {
        heap.alloc(Object::Bare, &mut (), &mut allocation).unwrap();
    }
    let mut resources = Resources::new(Limits {
        cpu: 2,
        ..Limits::unlimited()
    });
    assert!(heap.collect_young(&mut (), None, &mut resources).is_err());
    assert_eq!(
        resources.stop_reason(),
        Some(crate::resources::StopReason::CpuExhausted)
    );
}

#[test]
fn identity_is_stable_across_moves_and_distinct_per_object() {
    let mut heap = Heap::default();
    let mut resources = unlimited();
    let first = string(&mut heap, &mut (), &mut resources, "a");
    let second = string(&mut heap, &mut (), &mut resources, "b");
    let before = heap.identity(first).unwrap().unwrap();
    assert_ne!(before, heap.identity(second).unwrap().unwrap());
    assert_eq!(heap.identity(Value::Int(1)).unwrap(), None);
    heap.collect_young(&mut (), None, &mut resources).unwrap();
    assert_eq!(heap.identity(first).unwrap(), Some(before));
    let alias = heap.handle(&heap.store(first));
    assert!(heap.identical(first, alias));
    assert!(!heap.identical(first, second));
}

#[test]
fn replacing_a_payload_keeps_identity_and_adjusts_accounting() {
    let mut heap = Heap::default();
    let mut resources = unlimited();
    let value = heap
        .alloc(Object::Bytes(vec![0; 100]), &mut (), &mut resources)
        .unwrap();
    let identity = heap.identity(value).unwrap();
    let before = resources.outcome(0, 0, 0).usage.memory_current;
    heap.replace_payload(value, Object::Bytes(vec![0; 10]), &mut (), &mut resources)
        .unwrap();
    assert_eq!(resources.outcome(0, 0, 0).usage.memory_current, before - 90);
    assert_eq!(heap.identity(value).unwrap(), identity);
    heap.replace_payload(value, Object::Bytes(vec![0; 300]), &mut (), &mut resources)
        .unwrap();
    assert_eq!(heap.object_bytes(value).unwrap(), OBJECT_HEADER + 300);
}

#[test]
fn cloned_heaps_diverge_cleanly() {
    let mut heap = Heap::default();
    let mut resources = unlimited();
    let mut roots: Vec<Ref> = Vec::new();
    let value = list(&mut heap, &mut roots, &mut resources, &[]);
    roots.push(heap.store(value));
    heap.truncate_handles(0);
    let mut cloned = heap.clone();
    let mut cloned_roots = vec![roots[0].dup()];
    let original = heap.handle(&roots[0]);
    heap.modify(original, |_, object| {
        let Object::List(items) = object else {
            unreachable!()
        };
        items.push(Ref(Raw::int(1)));
    })
    .unwrap();
    let copy = cloned.handle(&cloned_roots[0]);
    assert!(matches!(cloned.get(copy).unwrap(), Object::List(items) if items.is_empty()));
    cloned
        .collect_full(&mut cloned_roots, None, &mut resources)
        .unwrap();
    assert_eq!(cloned.live_objects(), 1);
}

/// Every heap slot is as large as the largest `Object` variant and every object is charged
/// `OBJECT_HEADER` for it, so variants with big inline payloads are boxed to keep small
/// objects cheap for guests.
#[test]
fn heap_slots_stay_small() {
    assert!(std::mem::size_of::<Object>() <= 64);
    assert!(std::mem::size_of::<HeapObject>() <= 80);
    assert_eq!(
        std::mem::size_of::<Option<HeapObject>>() as u64,
        OBJECT_HEADER
    );
}
