import logging

import pytest

from tests.testmodels import (
    Car,
    Parking,
    Truck,
    Vehicle,
    VehicleLabel,
    VehicleOwner,
)
from tortoise import fields
from tortoise.contrib.test import requireCapability
from tortoise.exceptions import (
    ConfigurationError,
    IntegrityError,
    OperationalError,
    ParamsError,
    UnSupportedError,
)
from tortoise.expressions import F, Q
from tortoise.functions import Count, Upper
from tortoise.models import Model, _init_polymorphic_models
from tortoise.query_utils import Prefetch
from tortoise.queryset import load_subtypes


@pytest.mark.asyncio
async def test_declaration(db):
    """A subtype's key is a one-to-one to its parent; the parent's fields are reached
    through it."""
    meta = Car._meta
    assert meta.parent is Vehicle
    assert meta.parent_link == "vehicle_ptr"
    assert meta.pk_attr == "vehicle_ptr_id"
    assert meta.fields_db_projection["vehicle_ptr_id"] == "id"
    assert {"name", "kind", "owner", "owner_id", "labels"} <= set(meta.inherited)
    assert meta.inherited["id"] == "vehicle_ptr_id"
    assert meta.inherited["name"] == "vehicle_ptr__name"
    assert "name" not in meta.fields_map
    # The parent's links to its subtypes are not inherited.
    assert "car" not in meta.inherited and "truck" not in meta.inherited
    assert Vehicle._meta.subtypes == {"car": Car, "truck": Truck}
    assert Truck._meta.parent_link == "vehicle"
    assert Truck._meta.inherited["id"] == "vehicle_id"


@pytest.mark.asyncio
async def test_create(db):
    owner = await VehicleOwner.create(name="ada")
    car = await Car.create(name="beetle", color="red", owner=owner)
    assert isinstance(car, Vehicle)
    assert car.pk == car.id == car.vehicle_ptr_id
    assert (car.name, car.kind, car.color, car.owner_id) == ("beetle", "car", "red", owner.id)
    parent = await Vehicle.get(id=car.id)
    assert (parent.name, parent.kind) == ("beetle", "car")
    assert await Car.filter(id=car.id).values_list("color", flat=True) == ["red"]

    plain = await Vehicle.create(name="cart")
    assert plain.kind == "vehicle"
    truck = await Truck.create(name="big", payload=10, kind="ignored")
    assert truck.kind == "truck"
    assert truck.pk == truck.vehicle_id


@requireCapability(supports_transactions=True)
@pytest.mark.asyncio
async def test_create_rolls_back_the_parent_row(db_truncate):
    """Outside a transaction, a subtype's rows are written in one."""
    await Car.create(name="a", color="red", plate="X1")
    with pytest.raises(IntegrityError):
        await Car.create(name="b", color="red", plate="X1")
    assert await Vehicle.all().values_list("name", flat=True) == ["a"]


@pytest.mark.asyncio
async def test_query_inherited_fields(db):
    owner = await VehicleOwner.create(name="ada")
    beetle = await Car.create(name="beetle", color="red", owner=owner)
    await Car.create(name="mini", color="blue")
    await Truck.create(name="big", payload=10, owner=owner)

    assert await Car.filter(name="beetle").first() == beetle
    loaded = await Car.get(id=beetle.id)
    assert (loaded.name, loaded.kind, loaded.color) == ("beetle", "car", "red")
    assert [c.name for c in await Car.all().order_by("-name")] == ["mini", "beetle"]
    assert [c.name for c in await Car.exclude(name="mini")] == ["beetle"]
    assert await Car.filter(Q(name="mini") | Q(color="red")).count() == 2
    assert await Car.filter(name__icontains="EET", color="red").count() == 1
    assert await Car.filter(owner__name="ada").count() == 1
    assert await Car.filter(owner=owner).count() == 1
    assert await Car.filter(id__in=[beetle.id]).exists()
    assert await Car.filter(name=F("name")).count() == 2
    assert await Car.all().order_by("name").values("id", "name", "color", "owner__name") == [
        {"id": beetle.id, "name": "beetle", "color": "red", "owner__name": "ada"},
        {"id": beetle.id + 1, "name": "mini", "color": "blue", "owner__name": None},
    ]
    assert await Car.filter(color="red").values() == [
        {
            "id": beetle.id,
            "name": "beetle",
            "kind": "car",
            "owner_id": owner.id,
            "color": "red",
            "plate": None,
        }
    ]
    assert await Car.all().order_by("name").values_list("name", flat=True) == ["beetle", "mini"]
    assert await Car.filter(color="red").values_list() == [
        (beetle.id, "beetle", "car", owner.id, "red", None)
    ]
    assert await Car.all().annotate(n=Count("id")).group_by("kind").values("kind", "n") == [
        {"kind": "car", "n": 2}
    ]
    # The parent's default identity column filters like any other.
    assert await Vehicle.filter(kind="car").count() == 2
    # A subtype's own fields, from the parent.
    assert await Vehicle.filter(car__color="red").values_list("name", flat=True) == ["beetle"]
    # Its subtype's default ordering on an inherited field.
    await Truck.create(name="axle", payload=1)
    assert [t.name for t in await Truck.all()] == ["axle", "big"]


@pytest.mark.asyncio
async def test_query_sql_joins_the_parent_once(db):
    sql = Car.filter(name="beetle").order_by("name").sql()
    assert sql.count("JOIN") == 1


@pytest.mark.asyncio
async def test_inherited_relations(db):
    owner = await VehicleOwner.create(name="ada")
    label = await VehicleLabel.create(name="old")
    car = await Car.create(name="beetle", color="red", owner=owner)
    await car.labels.add(label)

    loaded = await Car.get(id=car.id).prefetch_related("labels", "owner")
    assert [lb.name for lb in loaded.labels] == ["old"]
    assert loaded.owner.name == "ada"
    loaded = await Car.get(id=car.id)
    await loaded.fetch_related("labels")
    assert [lb.name for lb in loaded.labels] == ["old"]
    loaded = await Car.get(id=car.id).select_related("owner")
    assert loaded.owner.name == "ada"
    assert (await Car.get(id=car.id).only("id", "name")).name == "beetle"
    assert await Car.filter(labels__name="old").count() == 1
    assert await VehicleOwner.annotate(n=Count("vehicles")).values("name", "n") == [
        {"name": "ada", "n": 1}
    ]
    assert await label.vehicles.all().values_list("name", flat=True) == ["beetle"]


@pytest.mark.asyncio
async def test_prefetch_through_the_parent_reuses_its_rows(db, caplog):
    owner = await VehicleOwner.create(name="ada")
    label = await VehicleLabel.create(name="old")
    car = await Car.create(name="beetle", color="red", owner=owner)
    await car.labels.add(label)
    caplog.set_level(logging.DEBUG, logger="tortoise.db_client")
    caplog.clear()
    (loaded,) = await Car.all().prefetch_related("labels", "owner")
    # The parent rows are read once, joined to the cars.
    queries = [r.getMessage() for r in caplog.records if "SELECT" in r.getMessage()]
    assert [q for q in queries if 'FROM "vehicle"' in q] == []
    assert sum('JOIN "vehicle"' in q for q in queries) == 1
    assert ([lb.name for lb in loaded.labels], loaded.owner.name) == (["old"], "ada")


@pytest.mark.asyncio
async def test_foreign_key_to_subtype(db):
    car = await Car.create(name="beetle", color="red")
    parking = await Parking.create(car=car)
    assert parking.car_id == car.id
    loaded = await Parking.get(id=parking.id).prefetch_related("car")
    assert (loaded.car.name, loaded.car.color) == ("beetle", "red")
    assert await Parking.filter(car__name="beetle").count() == 1
    assert [p.id for p in await car.parkings] == [parking.id]


@pytest.mark.asyncio
async def test_save(db):
    car = await Car.create(name="beetle", color="red")
    car.name = "beetle 2"
    car.color = "blue"
    await car.save()
    assert await Car.filter(id=car.id).values("name", "color") == [
        {"name": "beetle 2", "color": "blue"}
    ]

    loaded = await Car.get(id=car.id)
    loaded.name, loaded.color = "beetle 3", "green"
    await loaded.save(update_fields=["name"])
    assert await Car.filter(id=car.id).values("name", "color") == [
        {"name": "beetle 3", "color": "blue"}
    ]
    loaded.update_from_dict({"name": "beetle 4", "color": "black"})
    await loaded.save(update_fields=["color"])
    assert await Car.filter(id=car.id).values("name", "color") == [
        {"name": "beetle 3", "color": "black"}
    ]

    await loaded.refresh_from_db()
    assert (loaded.name, loaded.color) == ("beetle 3", "black")

    clone = loaded.clone()
    clone.name = "copy"
    await clone.save()
    assert clone.id != loaded.id
    assert await Car.all().order_by("id").values_list("name", "color") == [
        ("beetle 3", "black"),
        ("copy", "black"),
    ]


@pytest.mark.asyncio
async def test_queryset_update(db):
    await Car.create(name="beetle", color="red")
    await Car.create(name="mini", color="red")
    # Filtering on a field that the update changes, in either table.
    assert await Car.filter(name="beetle").update(name="b", color="blue") == 1
    assert await Car.filter(color="red").update(name="m") == 1
    assert await Car.filter(name="nothing").update(name="x") == 0
    assert await Car.all().order_by("name").values("name", "color") == [
        {"name": "b", "color": "blue"},
        {"name": "m", "color": "red"},
    ]
    assert await Car.filter(name="b").update(color="green") == 1


@pytest.mark.asyncio
async def test_delete(db):
    car = await Car.create(name="beetle", color="red")
    await Car.create(name="mini", color="blue")
    await Truck.create(name="big", payload=10)

    await car.delete()
    assert await Car.filter(name="mini").delete() == 1
    assert await Car.all().count() == 0
    assert await Vehicle.all().values_list("name", flat=True) == ["big"]


@pytest.mark.asyncio
async def test_polymorphic(db):
    owner = await VehicleOwner.create(name="ada")
    car = await Car.create(name="beetle", color="red", owner=owner)
    truck = await Truck.create(name="big", payload=10)
    plain = await Vehicle.create(name="cart")

    rows = await Vehicle.all().order_by("id").prefetch_related("owner").polymorphic()
    assert [type(row) for row in rows] == [Car, Truck, Vehicle]
    assert rows == [car, truck, plain]
    assert (rows[0].name, rows[0].color, rows[0].owner.name) == ("beetle", "red", "ada")
    assert rows[1].payload == 10

    assert type(await Vehicle.get(id=truck.id).polymorphic()) is Truck
    only_cars = await Vehicle.all().order_by("id").polymorphic(Car)
    assert [type(row) for row in only_cars] == [Car, Vehicle, Vehicle]
    rows = await load_subtypes(await Vehicle.all().order_by("id"), [Truck])
    assert [type(row) for row in rows] == [Vehicle, Truck, Vehicle]
    assert await load_subtypes([]) == []
    assert await Vehicle.filter(name="none").polymorphic() == []
    with pytest.raises(ValueError):
        await Vehicle.all().only("id").polymorphic()
    # Without it, parent rows stay parent instances.
    assert {type(row) for row in await Vehicle.all()} == {Vehicle}


@pytest.mark.asyncio
async def test_unsupported(db):
    with pytest.raises(UnSupportedError):
        await Car.bulk_create([Car(name="a", color="red")])
    car = await Car.create(name="a", color="red")
    with pytest.raises(UnSupportedError):
        await Car.bulk_update([car], fields=["name"])
    car.color = "blue"
    await Car.bulk_update([car], fields=["color"])
    assert (await Car.get(id=car.id)).color == "blue"
    with pytest.raises(ParamsError):
        Car.all().polymorphic()


@pytest.mark.asyncio
async def test_parent_row_not_loaded(db):
    await Car.create(name="beetle", color="red")
    (car,) = await Car.raw('SELECT * FROM "car"')
    assert car.color == "red"
    with pytest.raises(AttributeError, match="select_related"):
        car.name
    with pytest.raises(OperationalError, match="select_related"):
        car.update_from_dict({"name": "renamed"})
    car.update_from_dict({"color": "blue"})
    await car.save()
    assert (await Car.get(id=car.id)).color == "blue"


def test_one_level_only() -> None:
    with pytest.raises(ConfigurationError, match="one level"):

        class SportsCar(Car):
            class Meta:
                polymorphic_identity = "sports"

    with pytest.raises(ConfigurationError, match="one level"):

        class Bike(Vehicle):
            class Meta:
                polymorphic_on = "kind"


def test_subtype_key_is_its_link() -> None:
    with pytest.raises(ConfigurationError, match="primary key"):

        class Bike(Vehicle):
            code = fields.IntField(primary_key=True)


def test_ordering_inherited_from_parent() -> None:
    class Ordered(Model):
        name = fields.CharField(max_length=10)
        kind = fields.CharField(max_length=10)

        class Meta:
            polymorphic_on = "kind"
            ordering = ["-name"]

    class Sub(Ordered):
        class Meta:
            polymorphic_identity = "sub"

    _init_polymorphic_models([Ordered, Sub])
    assert Sub._meta.ordering == Ordered._meta.ordering
    assert Sub._meta.lookup("name") == "ordered_ptr__name"


def test_hierarchy_checks() -> None:
    class Base(Model):
        kind = fields.CharField(max_length=10)

        class Meta:
            polymorphic_on = "kind"

    class NoIdentity(Base):
        pass

    with pytest.raises(ConfigurationError, match="polymorphic_identity"):
        _init_polymorphic_models([Base, NoIdentity])

    class First(Base):
        class Meta:
            polymorphic_identity = "x"

    class Second(Base):
        class Meta:
            polymorphic_identity = "x"

    with pytest.raises(ConfigurationError, match="Duplicate polymorphic_identity"):
        _init_polymorphic_models([Base, First, Second])

    class Shadowing(Base):
        kind = fields.CharField(max_length=10)

        class Meta:
            polymorphic_identity = "y"

    with pytest.raises(ConfigurationError, match="shadows"):
        _init_polymorphic_models([Base, Shadowing])

    class NoColumn(Model):
        class Meta:
            polymorphic_on = "missing"

    class Sub(NoColumn):
        class Meta:
            polymorphic_identity = "z"

    with pytest.raises(ConfigurationError, match="polymorphic_on"):
        _init_polymorphic_models([NoColumn, Sub])


@requireCapability(supports_transactions=True)
@pytest.mark.asyncio
async def test_select_for_update(db):
    car = await Car.create(name="beetle", color="red")
    locked = await Car.filter(id=car.id).select_for_update().get()
    assert (locked.name, locked.color) == ("beetle", "red")
    assert await Car.filter(name="beetle").select_for_update().count() == 1


@pytest.mark.asyncio
async def test_partial_subtype_saves_its_parent_fields(db):
    car = await Car.create(name="beetle", color="red")
    partial = await Car.filter(id=car.id).only("id", "name").get()
    partial.name = "renamed"
    await partial.save(update_fields=["name"])
    assert await Vehicle.filter(id=car.id).values_list("name", flat=True) == ["renamed"]


@pytest.mark.asyncio
async def test_annotation_named_as_a_parent_field(db):
    await Car.create(name="beetle", color="red")
    await Car.create(name="mini", color="blue")
    rows = await Car.annotate(name=Upper("color")).order_by("name")
    assert [row.color for row in rows] == ["blue", "red"]


@pytest.mark.asyncio
async def test_subtype_reached_through_select_related(db):
    car = await Car.create(name="beetle", color="red")
    parking = await Parking.create(car=car)
    loaded = await Parking.filter(id=parking.id).select_related("car").get()
    assert (loaded.car.name, loaded.car.color) == ("beetle", "red")
    partial = await Parking.filter(id=parking.id).only("id", "car__name").get()
    assert partial.car.name == "beetle"
    # A parent relation on top: its columns are selected once.
    sql = Car.all().select_related("owner").sql()
    assert sql.count('"car__vehicle_ptr"."name"') == 1


@pytest.mark.asyncio
async def test_queryset_writes_in_chunks(db, monkeypatch):
    monkeypatch.setattr("tortoise.queryset._SUBTYPE_KEYS_PER_STATEMENT", 2)
    for name in "abcde":
        await Car.create(name=name, color="red")
    assert await Car.filter(color="red").update(name="x", color="blue") == 5
    assert await Car.filter(name="x", color="blue").count() == 5
    assert await Car.all().delete() == 5
    assert await Vehicle.all().count() == 0


@pytest.mark.asyncio
async def test_update_from_another_table(db):
    await Car.create(name="beetle", color="red")
    with pytest.raises(UnSupportedError):
        await Car.all().update(color=F("name"))
    await Car.all().update(plate=F("color"))
    assert await Car.all().values_list("plate", flat=True) == ["red"]


@pytest.mark.asyncio
async def test_polymorphic_keeps_annotations_and_to_attr(db):
    owner = await VehicleOwner.create(name="ada")
    label = await VehicleLabel.create(name="old")
    car = await Car.create(name="beetle", color="red", owner=owner)
    await car.labels.add(label)
    (row,) = await Vehicle.annotate(n=Count("labels")).polymorphic()
    assert type(row) is Car and row.n == 1
    (loaded,) = await Car.all().prefetch_related(
        Prefetch("owner", VehicleOwner.all(), to_attr="the_owner")
    )
    assert loaded.the_owner.name == "ada"


@pytest.mark.asyncio
async def test_construct(db):
    car = Car.construct(id=3, name="beetle", color="red")
    assert (car.id, car.name, car.kind, car.color) == (3, "beetle", "car", "red")
    assert getattr(car, "vehicle_ptr").name == "beetle"
