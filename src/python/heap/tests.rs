//! Heap tests drive allocation and collection directly, with explicit root sets and pins, so
//! they can force young and full collections and observe reclamation, slot reuse and
//! accounting.

use super::*;
use crate::python::error::PyErrorKind;
use crate::resources::{Limits, Resources};

fn unlimited() -> Resources {
    Resources::new(Limits::unlimited())
}

fn list(heap: &mut Heap, roots: &dyn Roots, resources: &mut Resources, items: &[Value]) -> Value {
    heap.alloc(
        Object::List(Ref::all(items.iter().copied())),
        roots,
        resources,
    )
    .unwrap()
}

fn string(heap: &mut Heap, roots: &dyn Roots, resources: &mut Resources, text: &str) -> Value {
    heap.alloc(Object::String(text.into()), roots, resources)
        .unwrap()
}

fn text(heap: &Heap, value: Value) -> String {
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
    let value = list(&mut heap, &(), &mut resources, &items);
    assert_eq!(
        heap.object_bytes(value).unwrap(),
        OBJECT_HEADER + 10 * MODELED_VALUE_BYTES
    );
}

#[test]
fn pinned_values_survive_a_young_collection_in_place() {
    let mut heap = Heap::default();
    let mut resources = unlimited();
    let kept = string(&mut heap, &(), &mut resources, "kept");
    let dropped = string(&mut heap, &(), &mut resources, "dropped");
    let dropped_slot = Ref::from(dropped);
    heap.truncate_pins(1, &mut resources);
    assert_eq!(heap.live_objects(), 2);

    heap.collect_young(&(), None, &mut resources).unwrap();
    assert_eq!(heap.live_objects(), 1);
    // The surviving value was not rewritten: objects never move.
    assert_eq!(text(&heap, kept), "kept");
    assert_eq!(heap.stats().promoted_objects, 1);
    assert_eq!(
        heap.get(heap.value(&dropped_slot)).unwrap_err().message(),
        "stale reference to a freed object"
    );
}

#[test]
fn a_reused_slot_does_not_answer_for_its_previous_object() {
    let mut heap = Heap::default();
    let mut resources = unlimited();
    let first = string(&mut heap, &(), &mut resources, "first");
    let stale = Ref::from(first);
    heap.truncate_pins(0, &mut resources);
    heap.collect_young(&(), None, &mut resources).unwrap();
    let second = string(&mut heap, &(), &mut resources, "second");
    assert_eq!(
        Heap::object_id(second).unwrap().index(),
        Heap::object_id(first).unwrap().index(),
        "the freed slot is reused"
    );
    assert!(!second.is_ref(&stale));
    assert_eq!(
        heap.get(heap.value(&stale)).unwrap_err().message(),
        "stale reference to a freed object"
    );
    assert_eq!(text(&heap, second), "second");
}

#[test]
fn reported_roots_keep_their_targets() {
    let mut heap = Heap::default();
    let mut resources = unlimited();
    let mut roots: Vec<Ref> = Vec::new();
    let value = string(&mut heap, &roots, &mut resources, "rooted");
    roots.push(Ref::from(value));
    heap.truncate_pins(0, &mut resources);

    heap.collect_young(&roots, None, &mut resources).unwrap();
    let value = heap.value(&roots[0]);
    assert_eq!(text(&heap, value), "rooted");
    assert_eq!(heap.live_objects(), 1);
}

#[test]
fn references_stored_into_old_objects_are_remembered() {
    let mut heap = Heap::default();
    let mut resources = unlimited();
    let container = list(&mut heap, &(), &mut resources, &[]);
    heap.collect_young(&(), None, &mut resources).unwrap();
    // `container` is old now; store a young string into it without any other root.
    let young = string(&mut heap, &(), &mut resources, "young");
    heap.modify(container, |object| {
        let Object::List(items) = object else {
            unreachable!()
        };
        items.push(Ref::from(young));
    })
    .unwrap();
    heap.truncate_pins(1, &mut resources);

    heap.collect_young(&(), None, &mut resources).unwrap();
    let Object::List(items) = heap.get(container).unwrap() else {
        unreachable!()
    };
    let element = heap.value(&items[0]);
    assert_eq!(text(&heap, element), "young");
}

#[test]
fn full_collection_reclaims_unreachable_cycles_and_releases_memory() {
    let mut heap = Heap::default();
    let mut resources = unlimited();
    let first = list(&mut heap, &(), &mut resources, &[]);
    let second = list(&mut heap, &(), &mut resources, &[first]);
    heap.reserve_object_growth(first, MODELED_VALUE_BYTES, &(), &mut resources)
        .unwrap();
    heap.modify(first, |object| {
        let Object::List(items) = object else {
            unreachable!()
        };
        items.push(Ref::from(second));
    })
    .unwrap();

    // Both are pinned.
    heap.collect_full(&(), None, &mut resources).unwrap();
    assert_eq!(heap.live_objects(), 2);
    heap.truncate_pins(0, &mut resources);
    let released = heap.collect_full(&(), None, &mut resources).unwrap();
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
    let element = string(&mut heap, &(), &mut resources, "element");
    let element_slot = Ref::from(element);
    // Fill the young space with empty byte objects until less than one of them fits in the
    // budget. A one-element list costs more than an empty object, so allocating the container is
    // what triggers the collection.
    let before = heap.young_bytes;
    heap.alloc(Object::Bytes(Vec::new()), &(), &mut resources)
        .unwrap();
    let garbage_cost = heap.young_bytes - before;
    while heap.young_bytes + garbage_cost <= gc::young_budget(limit) {
        heap.alloc(Object::Bytes(Vec::new()), &(), &mut resources)
            .unwrap();
        heap.truncate_pins(0, &mut resources);
    }
    let minor_before = heap.stats().minor_collections;
    // Only the pending list names `element` now.
    let container = heap
        .alloc(Object::List(vec![element_slot]), &(), &mut resources)
        .unwrap();
    assert_eq!(
        heap.stats().minor_collections,
        minor_before + 1,
        "allocation collected the young space"
    );
    assert!(heap.young.len() <= 2);
    let Object::List(items) = heap.get(container).unwrap() else {
        unreachable!()
    };
    let element = heap.value(&items[0]);
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
        heap.alloc(Object::Bytes(vec![0; 1024]), &(), &mut resources)
            .unwrap();
        heap.truncate_pins(0, &mut resources);
    }
    assert!(heap.stats().major_collections > 0);
    // Live data beyond the limit is still an error.
    let mut roots: Vec<Ref> = Vec::new();
    let error = loop {
        match heap.alloc(Object::Bytes(vec![0; 1024]), &roots, &mut resources) {
            Ok(value) => roots.push(Ref::from(value)),
            Err(error) => break error,
        }
        heap.truncate_pins(0, &mut resources);
    };
    assert_eq!(error.kind(), Some(&PyErrorKind::Resource));
    assert!(roots.len() >= 8 && roots.len() < 16);
}

#[test]
fn collection_charges_cpu_for_every_object() {
    let mut heap = Heap::default();
    let mut allocation = unlimited();
    for _ in 0..3 {
        heap.alloc(Object::Bare, &(), &mut allocation).unwrap();
    }
    let mut resources = Resources::new(Limits {
        cpu: 2,
        ..Limits::unlimited()
    });
    assert!(heap.collect_young(&(), None, &mut resources).is_err());
    assert_eq!(
        resources.stop_reason(),
        Some(crate::resources::StopReason::CpuExhausted)
    );
}

#[test]
fn identity_is_stable_across_collections_and_distinct_per_object() {
    let mut heap = Heap::default();
    let mut resources = unlimited();
    let first = string(&mut heap, &(), &mut resources, "a");
    let second = string(&mut heap, &(), &mut resources, "b");
    let before = heap.identity(first).unwrap().unwrap();
    assert_ne!(before, heap.identity(second).unwrap().unwrap());
    assert_eq!(heap.identity(Value::Int(1)).unwrap(), None);
    heap.collect_young(&(), None, &mut resources).unwrap();
    assert_eq!(heap.identity(first).unwrap(), Some(before));
    let alias = heap.value(&Ref::from(first));
    assert!(first.is(alias));
    assert!(!first.is(second));
}

#[test]
fn replacing_a_payload_keeps_identity_and_adjusts_accounting() {
    let mut heap = Heap::default();
    let mut resources = unlimited();
    let value = heap
        .alloc(Object::Bytes(vec![0; 100]), &(), &mut resources)
        .unwrap();
    let identity = heap.identity(value).unwrap();
    let before = resources.outcome(0, 0, 0).usage.memory_current;
    heap.replace_payload(value, Object::Bytes(vec![0; 10]), &(), &mut resources)
        .unwrap();
    assert_eq!(resources.outcome(0, 0, 0).usage.memory_current, before - 90);
    assert_eq!(heap.identity(value).unwrap(), identity);
    heap.replace_payload(value, Object::Bytes(vec![0; 300]), &(), &mut resources)
        .unwrap();
    assert_eq!(heap.object_bytes(value).unwrap(), OBJECT_HEADER + 300);
}

#[test]
fn cloned_heaps_diverge_cleanly() {
    let mut heap = Heap::default();
    let mut resources = unlimited();
    let mut roots: Vec<Ref> = Vec::new();
    let value = list(&mut heap, &roots, &mut resources, &[]);
    roots.push(Ref::from(value));
    heap.truncate_pins(0, &mut resources);
    let mut cloned = heap.clone();
    let cloned_roots = vec![roots[0].dup()];
    let original = heap.value(&roots[0]);
    heap.modify(original, |object| {
        let Object::List(items) = object else {
            unreachable!()
        };
        items.push(Ref(Raw::int(1)));
    })
    .unwrap();
    let copy = cloned.value(&cloned_roots[0]);
    assert!(matches!(cloned.get(copy).unwrap(), Object::List(items) if items.is_empty()));
    cloned.truncate_pins(0, &mut resources);
    cloned
        .collect_full(&cloned_roots, None, &mut resources)
        .unwrap();
    assert_eq!(cloned.live_objects(), 1);
}

#[test]
fn releasing_pins_releases_only_the_scratch_reserved_after_them() {
    let mut heap = Heap::default();
    let mut resources = Resources::new(Limits {
        memory: 1000,
        ..Limits::unlimited()
    });
    let kept = string(&mut heap, &(), &mut resources, "kept");
    let kept_bytes = resources.memory_mark();
    heap.reserve_scratch(100, &mut resources).unwrap();
    // A nested scope opens here, after the outer reservation, and reserves its own.
    let nested = heap.pin_count();
    heap.reserve_scratch(200, &mut resources).unwrap();
    assert_eq!(resources.memory_mark(), kept_bytes + 300);

    // The markers name no object, so a collection keeps only the pinned string.
    heap.collect_full(&(), None, &mut resources).unwrap();
    assert_eq!(text(&heap, kept), "kept");
    heap.truncate_pins(nested, &mut resources);
    assert_eq!(resources.memory_mark(), kept_bytes + 100);
    heap.truncate_pins(0, &mut resources);
    heap.collect_full(&(), None, &mut resources).unwrap();
    assert_eq!(resources.memory_mark(), 0);
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
