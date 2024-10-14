from astropy.time import Time
import pytz
from datetime import datetime
import astropy.units as u
from astroplan import EclipsingSystem
from astropy.coordinates import SkyCoord
from astroplan import FixedTarget, Observer, EclipsingSystem
from astroplan import (PrimaryEclipseConstraint, is_event_observable,
                       AtNightConstraint, AltitudeConstraint, LocalTimeConstraint)
import datetime as dt

### No need to change for BoiseState
boiseState = Observer(longitude=-116.208710*u.deg, latitude=43.602*u.deg,
                  elevation=821*u.m, name="BoiseState", timezone="US/Mountain")
# elevation?

coords = SkyCoord(ra=268.0291*u.deg, dec=37.54633*u.deg)
TrES_3 = FixedTarget(coord=coords, name="TrES-3")

#### Change value to mid transit time.
primary_eclipse_time = Time(2460473.89324, format='jd')  

### Change orbital period in days
orbital_period = 1.30618581 * u.day

### Change to duration in days with a hour padded
eclipse_duration_padded = (0.05907+(1/12)) * u.day

### Change name to target
transit = EclipsingSystem(primary_eclipse_time=primary_eclipse_time,
                           orbital_period=orbital_period, duration=eclipse_duration_padded,
                           name='TrES-3')

### Change to sunset and sunrise
obs_time = Time('2024-10-01 00:00')
n_transits = 20
min_local_time = dt.time(18, 0) # Start time of transit
max_local_time = dt.time(5, 0) # End time of transit

midtransit_times = transit.next_primary_eclipse_time(obs_time, n_eclipses=n_transits)
# event times and time constraint have to be in the same time zone (utc)
constraints = [AltitudeConstraint(min=30*u.deg), LocalTimeConstraint(min=min_local_time, max=max_local_time)]
ing_egr = transit.next_primary_ingress_egress_time(obs_time, n_eclipses=n_transits)
ing_egr_bool = is_event_observable(constraints, boiseState, TrES_3, times_ingress_egress=ing_egr)
filtered_ing_egr = [time for time, is_observable in zip(ing_egr, ing_egr_bool[0]) if is_observable]

mdt = pytz.timezone('US/Mountain')
mdt_datetime = ing_egr.to_datetime(timezone=mdt)
ing_egr_bool = is_event_observable(constraints, boiseState, target, times_ingress_egress=ing_egr)
print(ing_egr_bool)
filtered_ing_egr = [time for time, is_observable in zip(mdt_datetime, ing_egr_bool[0]) if is_observable]

print(filtered_ing_egr)

# This is in JD/UTC, we want to change it to ISO/MDT

count = 0
for i in filtered_ing_egr:
    print(f"Eclipse {count+1}: Ingress: {i[0].iso}, Egress: {i[1].iso}")
    count += 1


next_eclipses = transit.next_primary_ingress_egress_time(obs_time, n_eclipses=20)

# Convert the ingress and egress times to ISO format
ingress_times_iso = [Time(t[0], format='jd').iso for t in next_eclipses]
egress_times_iso = [Time(t[1], format='jd').iso for t in next_eclipses]

# Print the results
# for i, (ingress, egress) in enumerate(zip(ingress_times_iso, egress_times_iso)):
#     print(f"Eclipse {i+1}: Padded_Ingress: {ingress}, Padded_Egress: {egress}")