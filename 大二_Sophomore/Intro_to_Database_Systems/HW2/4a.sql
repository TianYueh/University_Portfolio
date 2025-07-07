with A as(
	select Country_Others.CountryName, Continent_Name, Date, stringencyindex_average_fordisplay
	from country_and_continent_list join Country_others 
	using (CountryName)
),
B as(
	select continent_name, date, max(stringencyindex_average_fordisplay) as max
	from A
	where Date='20200601'
	or Date='20210601'
	or Date='20220601'
	group by Continent_Name, Date
)

select B.Continent_Name, CountryName, B.Date, B.max
from A join B on A.continent_name=B.continent_Name 
	and A.Date=B.Date
	and A.stringencyindex_average_fordisplay=B.max
order by Date, Continent_Name, CountryName