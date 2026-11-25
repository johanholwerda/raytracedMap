import { normalizeName } from './worldCities';

export interface NotableBuilding {
  name: string;
  city: string;
  country: string;
  lat: number;
  lng: number;
}

/**
 * Famous / impressive buildings, mostly ones that are mapped in a lot of detail
 * on OpenStreetMap (building:part roofs, towers, domes). Coordinates point at
 * the building itself so it ends up at the projection center.
 */
export const NOTABLE_BUILDINGS: NotableBuilding[] = [
  // --- Europe ---
  { name: 'Eiffel Tower',             city: 'Paris',          country: 'France',         lat: 48.85826,  lng: 2.29450 },
  { name: 'Notre-Dame de Paris',      city: 'Paris',          country: 'France',         lat: 48.85297,  lng: 2.34989 },
  { name: 'Sacré-Cœur',               city: 'Paris',          country: 'France',         lat: 48.88671,  lng: 2.34310 },
  { name: 'Mont-Saint-Michel',        city: 'Mont-Saint-Michel', country: 'France',      lat: 48.63600,  lng: -1.51140 },
  { name: 'Strasbourg Cathedral',     city: 'Strasbourg',     country: 'France',         lat: 48.58180,  lng: 7.75070 },
  { name: 'Château de Chambord',      city: 'Chambord',       country: 'France',         lat: 47.61610,  lng: 1.51700 },
  { name: 'Palace of Westminster',    city: 'London',         country: 'United Kingdom', lat: 51.49947,  lng: -0.12470 },
  { name: "St Paul's Cathedral",      city: 'London',         country: 'United Kingdom', lat: 51.51381,  lng: -0.09840 },
  { name: 'Tower Bridge',             city: 'London',         country: 'United Kingdom', lat: 51.50546,  lng: -0.07537 },
  { name: 'Edinburgh Castle',         city: 'Edinburgh',      country: 'United Kingdom', lat: 55.94860,  lng: -3.19990 },
  { name: 'Cologne Cathedral',        city: 'Cologne',        country: 'Germany',        lat: 50.94130,  lng: 6.95830 },
  { name: 'Reichstag',                city: 'Berlin',         country: 'Germany',        lat: 52.51862,  lng: 13.37618 },
  { name: 'Ulm Minster',              city: 'Ulm',            country: 'Germany',        lat: 48.39850,  lng: 9.99160 },
  { name: 'Neuschwanstein Castle',    city: 'Schwangau',      country: 'Germany',        lat: 47.55760,  lng: 10.74980 },
  { name: 'Rijksmuseum',              city: 'Amsterdam',      country: 'Netherlands',    lat: 52.35999,  lng: 4.88520 },
  { name: 'Atomium',                  city: 'Brussels',       country: 'Belgium',        lat: 50.89490,  lng: 4.34150 },
  { name: 'Sagrada Família',          city: 'Barcelona',      country: 'Spain',          lat: 41.40363,  lng: 2.17436 },
  { name: 'Seville Cathedral',        city: 'Seville',        country: 'Spain',          lat: 37.38590,  lng: -5.99310 },
  { name: 'Mezquita of Córdoba',      city: 'Córdoba',        country: 'Spain',          lat: 37.87890,  lng: -4.77940 },
  { name: 'Guggenheim Museum',        city: 'Bilbao',         country: 'Spain',          lat: 43.26870,  lng: -2.93400 },
  { name: "St. Peter's Basilica",     city: 'Vatican City',   country: 'Vatican City',   lat: 41.90220,  lng: 12.45390 },
  { name: 'Colosseum',                city: 'Rome',           country: 'Italy',          lat: 41.89021,  lng: 12.49223 },
  { name: 'Milan Cathedral',          city: 'Milan',          country: 'Italy',          lat: 45.46420,  lng: 9.19190 },
  { name: 'Florence Cathedral',       city: 'Florence',       country: 'Italy',          lat: 43.77310,  lng: 11.25600 },
  { name: "St Mark's Basilica",       city: 'Venice',         country: 'Italy',          lat: 45.43450,  lng: 12.33970 },
  { name: 'Leaning Tower of Pisa',    city: 'Pisa',           country: 'Italy',          lat: 43.72300,  lng: 10.39660 },
  { name: 'Prague Castle',            city: 'Prague',         country: 'Czechia',        lat: 50.09040,  lng: 14.40050 },
  { name: 'Hungarian Parliament',     city: 'Budapest',       country: 'Hungary',        lat: 47.50710,  lng: 19.04570 },
  { name: "St. Basil's Cathedral",    city: 'Moscow',         country: 'Russia',         lat: 55.75250,  lng: 37.62310 },
  { name: 'Hagia Sophia',             city: 'Istanbul',       country: 'Türkiye',        lat: 41.00860,  lng: 28.98020 },
  { name: 'Parthenon',                city: 'Athens',         country: 'Greece',         lat: 37.97150,  lng: 23.72670 },
  { name: 'Siena Cathedral',          city: 'Siena',          country: 'Italy',          lat: 43.31779,  lng: 11.32909 },
  { name: 'Burgos Cathedral',         city: 'Burgos',         country: 'Spain',          lat: 42.34073,  lng: -3.70456 },
  { name: 'Frauenkirche',             city: 'Dresden',        country: 'Germany',        lat: 51.05196,  lng: 13.74162 },
  { name: 'Berlin Cathedral',         city: 'Berlin',         country: 'Germany',        lat: 52.51908,  lng: 13.40109 },
  { name: 'Blackpool Tower',          city: 'Blackpool',      country: 'United Kingdom', lat: 53.81589,  lng: -3.05524 },
  { name: 'Hallgrímskirkja',          city: 'Reykjavík',      country: 'Iceland',        lat: 64.14180,  lng: -21.92671 },
  { name: 'Helsinki Cathedral',       city: 'Helsinki',       country: 'Finland',        lat: 60.17042,  lng: 24.95218 },
  { name: 'Palace of Culture and Science', city: 'Warsaw',    country: 'Poland',         lat: 52.23177,  lng: 21.00580 },
  { name: "St Andrew's Church",       city: 'Kyiv',           country: 'Ukraine',        lat: 50.45900,  lng: 30.51799 },
  { name: 'Alexander Nevsky Cathedral', city: 'Sofia',        country: 'Bulgaria',       lat: 42.69581,  lng: 23.33279 },
  { name: 'Blue Mosque',              city: 'Istanbul',       country: 'Türkiye',        lat: 41.00538,  lng: 28.97685 },
  { name: 'Cathedral of Christ the Saviour', city: 'Moscow',  country: 'Russia',         lat: 55.74452,  lng: 37.60551 },
  { name: 'Ostankino Tower',          city: 'Moscow',         country: 'Russia',         lat: 55.81973,  lng: 37.61166 },
  { name: 'Moscow State University',  city: 'Moscow',         country: 'Russia',         lat: 55.70291,  lng: 37.53076 },
  { name: 'Heart of Chechnya Mosque', city: 'Grozny',         country: 'Russia',         lat: 43.31792,  lng: 45.69391 },

  // --- Americas ---
  { name: 'Empire State Building',    city: 'New York',       country: 'United States',  lat: 40.74840,  lng: -73.98570 },
  { name: 'One World Trade Center',   city: 'New York',       country: 'United States',  lat: 40.71270,  lng: -74.01340 },
  { name: 'Chrysler Building',        city: 'New York',       country: 'United States',  lat: 40.75160,  lng: -73.97550 },
  { name: 'United States Capitol',    city: 'Washington, D.C.', country: 'United States', lat: 38.88990, lng: -77.00910 },
  { name: 'Lincoln Memorial',         city: 'Washington, D.C.', country: 'United States', lat: 38.88926, lng: -77.05021 },
  { name: 'Willis Tower',             city: 'Chicago',        country: 'United States',  lat: 41.87890,  lng: -87.63590 },
  { name: 'Space Needle',             city: 'Seattle',        country: 'United States',  lat: 47.62050,  lng: -122.34930 },
  { name: 'CN Tower',                 city: 'Toronto',        country: 'Canada',         lat: 43.64260,  lng: -79.38710 },
  { name: 'Parliament Hill',          city: 'Ottawa',         country: 'Canada',         lat: 45.42506,  lng: -75.69992 },
  { name: 'Palacio de Bellas Artes',  city: 'Mexico City',    country: 'Mexico',         lat: 19.43520,  lng: -99.14120 },

  // --- Africa & Middle East ---
  { name: 'Hassan II Mosque',         city: 'Casablanca',     country: 'Morocco',        lat: 33.60830,  lng: -7.63280 },
  { name: 'Dome of the Rock',         city: 'Jerusalem',      country: 'Israel',         lat: 31.77800,  lng: 35.23540 },
  { name: 'Burj Khalifa',             city: 'Dubai',          country: 'UAE',            lat: 25.19720,  lng: 55.27440 },
  { name: 'Sheikh Zayed Grand Mosque', city: 'Abu Dhabi',     country: 'UAE',            lat: 24.41280,  lng: 54.47490 },
  { name: 'Azadi Tower',              city: 'Tehran',         country: 'Iran',           lat: 35.69974,  lng: 51.33806 },
  { name: 'Milad Tower',              city: 'Tehran',         country: 'Iran',           lat: 35.74470,  lng: 51.37527 },

  // --- Asia ---
  { name: 'Taj Mahal',                city: 'Agra',           country: 'India',          lat: 27.17510,  lng: 78.04210 },
  { name: 'Forbidden City',           city: 'Beijing',        country: 'China',          lat: 39.91690,  lng: 116.39070 },
  { name: 'Shanghai Tower',           city: 'Shanghai',       country: 'China',          lat: 31.23565,  lng: 121.50125 },
  { name: 'Oriental Pearl Tower',     city: 'Shanghai',       country: 'China',          lat: 31.24195,  lng: 121.49526 },
  { name: 'International Finance Centre', city: 'Hong Kong',  country: 'China',          lat: 22.28530,  lng: 114.15880 },
  { name: 'Taipei 101',               city: 'Taipei',         country: 'Taiwan',         lat: 25.03360,  lng: 121.56450 },
  { name: 'Shwedagon Pagoda',         city: 'Yangon',         country: 'Myanmar',        lat: 16.79830,  lng: 96.14950 },
  { name: 'Angkor Wat',               city: 'Siem Reap',      country: 'Cambodia',       lat: 13.41250,  lng: 103.86700 },
  { name: 'Petronas Towers',          city: 'Kuala Lumpur',   country: 'Malaysia',       lat: 3.15790,   lng: 101.71160 },
  { name: 'Marina Bay Sands',         city: 'Singapore',      country: 'Singapore',      lat: 1.28260,   lng: 103.86080 },
  { name: 'Gardens by the Bay',       city: 'Singapore',      country: 'Singapore',      lat: 1.28200,   lng: 103.86394 },
  { name: 'Borobudur',                city: 'Magelang',       country: 'Indonesia',      lat: -7.60790,  lng: 110.20380 },
  { name: 'Gyeongbokgung Palace',     city: 'Seoul',          country: 'South Korea',    lat: 37.57960,  lng: 126.97700 },
  { name: 'Lotte World Tower',        city: 'Seoul',          country: 'South Korea',    lat: 37.51250,  lng: 127.10250 },
  { name: 'Tokyo Tower',              city: 'Tokyo',          country: 'Japan',          lat: 35.65860,  lng: 139.74540 },
];

let searchNames: string[] | undefined;

/** The notable buildings whose name matches the query: name prefix first, then word start, then anywhere. */
export function searchNotableBuildings(query: string, limit: number = 3): NotableBuilding[] {
  const q = normalizeName(query);
  if (!q) return [];
  searchNames ??= NOTABLE_BUILDINGS.map(b => normalizeName(b.name));
  const prefix: NotableBuilding[] = [];
  const word: NotableBuilding[] = [];
  const contains: NotableBuilding[] = [];
  for (let i = 0; i < NOTABLE_BUILDINGS.length; i++) {
    const name = searchNames[i];
    const at = name.indexOf(q);
    if (at < 0) continue;
    if (at === 0) prefix.push(NOTABLE_BUILDINGS[i]);
    else if (/[\s\-'(.]/.test(name[at - 1])) word.push(NOTABLE_BUILDINGS[i]);
    else contains.push(NOTABLE_BUILDINGS[i]);
  }
  return [...prefix, ...word, ...contains].slice(0, limit);
}
