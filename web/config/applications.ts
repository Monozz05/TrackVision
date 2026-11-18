import type { ErrorCode } from '@/types'

// The Applications page (decision 0016). Everything in this file is engineering judgement written by hand, not a measurement
// of any application: the ranks, the reasons, the look-ahead times and the choice of scene can all be changed here. Numbers
// never live in this file. They come from the live sweep of the scene chosen on the page.
export const JUDGEMENT_NOTE = 'The priorities and reasons here are engineering judgement. None of them was measured on this application. The numbers come from the named scene, not from a real system. Collision performance has not been measured, and no research gate is passed.'

export interface Priority { code: ErrorCode; why: string; inUse: string }
export interface Application {
  id: string
  name: string
  summary: string
  camera: string
  timeAhead: string
  needs: string
  metric: { choice: string; why: string; ade: string; fde: string }
  predictor: string
  scenes: { id: string; shows: string }[]
  nearest: string | null                   // says so where no footage of this kind exists in these datasets
  priorities: Priority[]                   // first to fix first
}

const SCENE = {
  'MOT17-02': 'A large open square, from a fixed camera',
  'MOT17-04': 'A busy street at night, from a high fixed camera',
  'MOT17-05': 'A street filmed from a moving platform',
  'MOT17-09': 'Outside a shopping mall, from a low fixed camera',
  'MOT17-10': 'A street at night, filmed from a moving camera',
  'MOT17-11': 'Walking through a shopping mall, camera facing forward',
  'MOT17-13': 'Filmed from a bus at a busy junction',
  ETH: 'People at a building entrance, seen from above',
  HOTEL: 'A street outside a hotel, seen from above',
  UNIV: 'A dense crowd of students, seen from above',
  ZARA1: 'A pavement outside a shop, seen from above',
  ZARA2: 'A pavement outside a shop, seen from above',
} as const
const scenes = (...ids: (keyof typeof SCENE)[]) => ids.map((id) => ({ id, shows: SCENE[id] }))

export const applications: Application[] = [
  {
    id: 'self-driving', name: 'Self-driving car',
    summary: 'A vehicle predicts where people will be, so it can plan its path and its braking.',
    camera: 'Moving, at street level', timeAhead: 'Long, up to the full forecast',
    needs: 'An accurate final position for every person near the path of the vehicle.',
    metric: { choice: 'FDE', why: 'The plan depends on where a person will be when the car gets there, so the last point of the forecast counts for more than the average.', ade: 'A higher ADE means the whole predicted path is off, so the car leaves too much room or too little.', fde: 'A higher FDE means the person\'s end position is wrong, which can move the braking decision either way.' },
    predictor: 'CV. It beat CA on every scene measured, and CA magnifies noise most at long range, which is the range this application relies on.',
    scenes: scenes('MOT17-13', 'MOT17-05', 'MOT17-10'), nearest: null,
    priorities: [
      { code: 'A', why: 'Position noise did the most damage to ADE and FDE on every scene measured, and the damage grows with time ahead, which is the range a car relies on.', inUse: 'The predicted direction and end point drift. The car can brake for someone who is not entering its path, or fail to slow for someone who is.' },
      { code: 'B', why: 'One missing point in a person\'s history loses their forecast here, and a person with no forecast cannot be planned around.', inUse: 'People partly hidden by parked cars or by each other drop out of the plan just when they matter most.' },
      { code: 'D', why: 'A switch moves a track onto a neighbour, so speed and direction jump. The measured rise is much smaller than for noise, but it lands on people close together near traffic.', inUse: 'A sudden false jump can look like someone stepping out, or can hide someone who really is.' },
      { code: 'C', why: 'A break removes a forecast until the new track has enough history. It ranks last because the person is still detected and the forecast returns.', inUse: 'The person\'s forecast disappears for about 3 s after each break, then comes back.' },
    ],
  },
  {
    id: 'driver-assistance', name: 'Driver-assistance warning',
    summary: 'A car warns the driver, or brakes briefly, when a person may step into its path.',
    camera: 'Moving, at street level', timeAhead: 'Short, the next couple of seconds',
    needs: 'Warnings the driver can trust, with few false alarms.',
    metric: { choice: 'ADE', why: 'A warning looks at the next couple of seconds, so error along the path matters more than the last point of the forecast.', ade: 'A higher ADE means the near path is wrong, which is what sets off a warning or fails to.', fde: 'A higher FDE matters less here, because the system acts well before the last forecast step.' },
    predictor: 'CV. It beat CA on every scene measured, and it needs only two points, so it responds quickly to a change in a person\'s path.',
    scenes: scenes('MOT17-05', 'MOT17-13'), nearest: null,
    priorities: [
      { code: 'A', why: 'CV takes its speed from the last two points, so even small noise gives a false speed. This was the largest measured damage on every scene.', inUse: 'False speeds set off false alarms, and drivers who get many false alarms stop trusting the system.' },
      { code: 'D', why: 'A switch makes a track jump sideways in one step, which looks exactly like someone darting out. It is rarer than noise but harder to filter out.', inUse: 'A single switch can trigger an emergency warning with nobody in danger.' },
      { code: 'B', why: 'A lost forecast means no warning for that person. It ranks third because this system mainly acts on people close to the car, who tend to be seen clearly.', inUse: 'A missed warning for someone appearing from behind a parked car.' },
      { code: 'C', why: 'A break removes the forecast for about 3 s, but the person is still detected, so the system still knows they are there.', inUse: 'The warning stays silent until the new track has enough history.' },
    ],
  },
  {
    id: 'delivery-robot', name: 'Delivery robot',
    summary: 'A small robot shares pavements and corridors with people and steers around them.',
    camera: 'Moving, low and slow', timeAhead: 'Short, because it replans all the time',
    needs: 'Accurate short paths for the people closest to it.',
    metric: { choice: 'ADE', why: 'The robot replans constantly, so the whole near path matters more than one point far ahead.', ade: 'A higher ADE means the robot steers around paths nobody is taking.', fde: 'A higher FDE matters less, because the robot replans long before that point is reached.' },
    predictor: 'CV. It beat CA on every scene measured. At walking pace, CV\'s straight-line guess is rarely far wrong over a short time.',
    scenes: scenes('MOT17-11', 'MOT17-05'), nearest: null,
    priorities: [
      { code: 'A', why: 'The robot steers from predicted paths, and noise was the largest measured damage. Close to people, a small position error is a large share of the gap to them.', inUse: 'The robot hesitates, swerves or stops for paths nobody is really taking.' },
      { code: 'B', why: 'Without a forecast the robot can only react to where a person is now, not where they are going.', inUse: 'It reacts late to people coming out of a doorway or from behind others.' },
      { code: 'C', why: 'People near a robot often pass in front of each other, which breaks tracks, and each break costs about 3 s without a forecast.', inUse: 'Forecasts vanish for the very people it is passing.' },
      { code: 'D', why: 'At walking pace a swap between two neighbours moves a forecast only a short way, so it matters least here.', inUse: 'The robot briefly follows the wrong person\'s path, then corrects itself.' },
    ],
  },
  {
    id: 'assistive-navigation', name: 'Assistive navigation',
    summary: 'A wearable device tells a person with low vision about people moving nearby.',
    camera: 'Worn, moving at walking pace', timeAhead: 'Short, the next moment or two',
    needs: 'A forecast for everyone close by, because the user cannot see them.',
    metric: { choice: 'ADE', why: 'What matters is whether someone will be close in the next moments, which the whole path shows better than one point far ahead.', ade: 'A higher ADE means the device says someone is coming close when they are not, or the reverse.', fde: 'A higher FDE matters less, because the device warns about the near future.' },
    predictor: 'CV. It beat CA on every scene measured. A worn camera shakes, which acts like noise, and CA magnifies noise far more.',
    scenes: scenes('MOT17-11', 'MOT17-10'), nearest: null,
    priorities: [
      { code: 'B', why: 'The user relies on the device completely, so a person with no forecast is a person they know nothing about. Even 5% missing costs about a third of forecasts here.', inUse: 'Someone walks into the user with no warning at all.' },
      { code: 'A', why: 'Noise can make a person seem to head towards the user or away from them. It did the most damage to forecasts, but a wrong warning is less serious here than no warning.', inUse: 'False warnings, or warnings that point the wrong way.' },
      { code: 'C', why: 'People pass close to a wearer and hide each other, which breaks tracks and costs about 3 s of forecasts each time.', inUse: 'Warnings about someone stop, then restart as if a new person had appeared.' },
      { code: 'D', why: 'A swap matters least, because the user needs to know someone is there, not which person it is.', inUse: 'The warning stays useful even if two people trade labels.' },
    ],
  },
  {
    id: 'drone', name: 'Drone monitoring',
    summary: 'A drone watches a wide area from above, for example at an event or during a search.',
    camera: 'High, looking down', timeAhead: 'Medium',
    needs: 'Keeping small people in view.',
    metric: { choice: 'ADE', why: 'The aim is to follow movement across an area, so the whole path matters more than one end point.', ade: 'A higher ADE means predicted paths stray from the real ones across the map.', fde: 'A higher FDE means an end point on the map is wrong, which matters most if the drone is sent to that point.' },
    predictor: 'CV. It beat CA on every scene measured. From high up, a small error in the picture is a large one on the ground, and CA magnifies it more.',
    scenes: scenes('ETH', 'HOTEL'),
    nearest: 'These datasets have no footage from a moving drone. ETH and HOTEL look down from a fixed point, which is the nearest match. Real drone footage also has the motion of the drone.',
    priorities: [
      { code: 'B', why: 'From above, people are only a few pixels wide, and they are the first thing a detector loses. Each loss here also costs a forecast.', inUse: 'People disappear from the map, and the operator sees a gap rather than a warning.' },
      { code: 'A', why: 'One pixel covers a long way on the ground from up high, so a small box error is a large position error, and noise did the most damage to forecasts.', inUse: 'Predicted paths wander and the operator cannot tell where people are heading.' },
      { code: 'C', why: 'A person lost from view often comes back with a new identity, which splits one path into several.', inUse: 'One person appears as several, so counts and paths break up.' },
      { code: 'D', why: 'Seen from above, people overlap less than from the ground, so swaps are less likely.', inUse: 'Two nearby paths trade places for a while.' },
    ],
  },
  {
    id: 'cctv', name: 'CCTV security',
    summary: 'Cameras in stations, streets and buildings, where an operator or software follows a person through the scene.',
    camera: 'Fixed and elevated', timeAhead: 'Medium, but following matters more than forecasting',
    needs: 'Following one person without losing them or swapping them for someone else.',
    metric: { choice: 'Neither on its own', why: 'ADE and FDE measure forecasts, not who is who. Identity switches barely move them, yet they are the main failure here, so identity measures matter more than either.', ade: 'A higher ADE makes forecasts of where someone is heading less reliable, which matters for handing over to the next camera.', fde: 'A higher FDE matters in the same way, for the last point before a person leaves the view.' },
    predictor: 'CV. It beat CA on every scene measured, but forecasts are a helper here. The tracker\'s identity is what this job depends on.',
    scenes: scenes('MOT17-04', 'ETH', 'HOTEL'), nearest: null,
    priorities: [
      { code: 'D', why: 'This is where ADE is the wrong guide. The measured rise is small, but following the wrong person is the worst failure this job can have.', inUse: 'An alert, a search or a piece of evidence gets attached to the wrong person.' },
      { code: 'C', why: 'A break splits one person\'s journey into two identities, which then have to be joined back together.', inUse: 'The operator loses the person and has to find them again.' },
      { code: 'B', why: 'A short gap is a small problem if the same identity carries on afterwards, and a bigger one if it breaks the track.', inUse: 'Brief holes in the record of where someone went.' },
      { code: 'A', why: 'Noise did the most damage to forecasts, but this job rarely needs exact positions, so it ranks last here despite that.', inUse: 'Boxes shake a little, and following still works.' },
    ],
  },
  {
    id: 'crowd', name: 'Crowd management',
    summary: 'Operators at stations, stadiums and events watch density and flow to keep crowds safe.',
    camera: 'Fixed and high, over dense crowds', timeAhead: 'Medium to long, because crowds build up over time',
    needs: 'Correct counts and overall flow.',
    metric: { choice: 'ADE', why: 'Flow depends on how many people move and where to, so the whole path matters more than one person\'s end point.', ade: 'A higher ADE blurs where the crowd is heading, though errors partly cancel out across many people.', fde: 'A higher FDE blurs where people will gather, which matters when planning ahead.' },
    predictor: 'CV. It beat CA on every scene measured. People in crowds change direction often, and CA turns each small change into a large forecast error.',
    scenes: scenes('UNIV', 'MOT17-04', 'MOT17-02'), nearest: null,
    priorities: [
      { code: 'B', why: 'In a dense crowd people hide each other, so detections go missing most where it matters, and every miss is someone not counted.', inUse: 'Density looks lower than it is at exactly the busiest places.' },
      { code: 'C', why: 'A broken track makes one person look like two, which distorts any count of people crossing a line.', inUse: 'Flow counts drift upwards.' },
      { code: 'A', why: 'Noise did the most damage to each forecast, but crowd figures average over many people, so random noise matters less than it does for one person.', inUse: 'Individual forecasts are poor, while overall flow stays usable.' },
      { code: 'D', why: 'A swap between neighbours changes neither the count nor the overall flow.', inUse: 'Almost no effect on crowd figures.' },
    ],
  },
  {
    id: 'junction', name: 'Smart traffic junction',
    summary: 'A junction adjusts its signals, or warns vehicles, when people are about to cross.',
    camera: 'Fixed, over the street', timeAhead: 'Long, up to the full forecast',
    needs: 'Knowing whether a person will enter the road.',
    metric: { choice: 'FDE', why: 'What counts is where a person ends up relative to the kerb, which is the last point of the forecast.', ade: 'A higher ADE means the path is off, which matters less than whether it ends on the road.', fde: 'A higher FDE can place a person on the wrong side of the kerb, which changes the decision.' },
    predictor: 'CV. It beat CA on every scene measured. Long look-ahead is where CA magnifies noise most.',
    scenes: scenes('ZARA1', 'ZARA2', 'MOT17-02'),
    nearest: 'These datasets have no signal-controlled junction from a fixed camera. ZARA1, ZARA2 and MOT17-02 show people on pavements and in open areas, which is the nearest match.',
    priorities: [
      { code: 'A', why: 'Near a kerb, a small error decides whether a forecast ends on the road. Noise did the most damage to FDE on every scene measured.', inUse: 'Signals change for people who are not crossing, or do not change for people who are.' },
      { code: 'B', why: 'Vehicles and queues hide people near the kerb, and each lost forecast is someone the junction does not plan for.', inUse: 'People waiting behind a van are not seen until they step out.' },
      { code: 'D', why: 'People wait close together at crossings, which is where swaps happen, and a swap moves a forecast to the wrong person.', inUse: 'A person who is waiting can inherit the path of someone who is crossing.' },
      { code: 'C', why: 'A break costs about 3 s of forecasts, but a crossing lasts longer than that, so the junction can catch up.', inUse: 'A short delay before the system reacts to a person again.' },
    ],
  },
  {
    id: 'retail', name: 'Retail analytics',
    summary: 'Shops and malls measure visits, time spent and paths through the space.',
    camera: 'Fixed, often low', timeAhead: 'Not the main need, because the job is measuring what happened',
    needs: 'One whole, unbroken visit per person.',
    metric: { choice: 'Neither on its own', why: 'Forecasts matter little here. Whole, unbroken tracks matter much more than ADE or FDE.', ade: 'A higher ADE has little effect, because nothing here acts on the forecast.', fde: 'A higher FDE has little effect, for the same reason.' },
    predictor: 'CV, if forecasts are used at all. It beat CA on every scene measured, and it is the simpler of the two.',
    scenes: scenes('MOT17-09', 'ZARA1', 'ZARA2'), nearest: null,
    priorities: [
      { code: 'C', why: 'A break turns one visit into two, which cuts the time spent and inflates the visitor count at once.', inUse: 'Reports show more visitors staying for less time than really happened.' },
      { code: 'D', why: 'A swap joins parts of two shoppers\' paths, so each one gets credit for places they never went.', inUse: 'Heat maps and paths mix up different people.' },
      { code: 'B', why: 'Short gaps do little harm if the identity carries on afterwards, though long ones split the visit.', inUse: 'Small holes in a person\'s path.' },
      { code: 'A', why: 'Noise did the most damage to forecasts, but aisle-level paths do not need fine accuracy, so it ranks last here despite that.', inUse: 'Paths look a little rough, and the figures stay usable.' },
    ],
  },
  {
    id: 'workplace', name: 'Workplace safety',
    summary: 'Cameras on a factory or warehouse floor warn when a person is near a forklift, a machine or a restricted area.',
    camera: 'Fixed and elevated', timeAhead: 'Short to medium',
    needs: 'Never missing a person near a hazard.',
    metric: { choice: 'ADE', why: 'The warning depends on whether someone will be in the danger area shortly, so the near path counts for more than the last point.', ade: 'A higher ADE puts people in or out of a danger area wrongly.', fde: 'A higher FDE matters less, because the warning has to come well before that point.' },
    predictor: 'CV. It beat CA on every scene measured. People on a work floor stop and turn often, and CA turns each change into a large forecast error.',
    scenes: scenes('MOT17-02', 'MOT17-04'),
    nearest: 'These datasets have no factory or warehouse footage. MOT17-02 and MOT17-04 are fixed cameras watching open areas, which is the nearest match.',
    priorities: [
      { code: 'B', why: 'A person with no forecast gets no warning, and that is the failure that causes harm here. Even 5% missing costs about a third of forecasts.', inUse: 'A worker hidden by racking gets no warning before stepping into a forklift\'s path.' },
      { code: 'A', why: 'Noise did the most damage to forecasts, and it can push a path into or out of a danger area. Too many false alarms get alerts ignored or switched off.', inUse: 'False alarms, or late ones.' },
      { code: 'D', why: 'A swap can move a person\'s forecast towards a hazard or away from one, though the measured rise is much smaller than for noise.', inUse: 'A warning goes to the wrong person\'s path.' },
      { code: 'C', why: 'A break silences forecasts for about 3 s, but the person\'s current position is still seen, so a rule based on position alone keeps working.', inUse: 'A short delay in forecast-based warnings.' },
    ],
  },
]

// What each error does to each predictor and measure. These follow from how CV, CA and the injectors work, and they hold for
// every application, so they are written once.
export const errorEffects: Record<ErrorCode, { cv: string; ca: string; ade: string; fde: string }> = {
  A: {
    cv: 'CV takes speed and direction from the last two points only. Noise on those two points becomes a false speed, which is carried through every step ahead.',
    ca: 'CA also estimates acceleration from the last three points, and its effect grows with the square of the time ahead. That is why CA magnifies noise far more than CV.',
    ade: 'Every step of the forecast moves away from the true path, so ADE rises at every noise level.',
    fde: 'The error keeps growing to the last step, so FDE ends up larger than ADE.',
  },
  B: {
    cv: 'CV needs only two points, but in this project a forecast needs a complete history of 8 points. One missing point loses that forecast.',
    ca: 'CA loses the same forecasts as CV, for the same reason.',
    ade: 'ADE does not change for the people still predicted, because their history is complete and their forecast is the clean one.',
    fde: 'FDE behaves like ADE: no change for people still predicted. The damage is the forecasts lost.',
  },
  C: {
    cv: 'A break inside the 8-point history ends that forecast. The new identity has no history yet, so CV has nothing to work from for about 3 s.',
    ca: 'CA loses the same forecasts as CV, for the same reason.',
    ade: 'ADE does not change for the people still predicted, because their history has no break in it.',
    fde: 'FDE behaves like ADE: no change for people still predicted. The damage is the forecasts lost.',
  },
  D: {
    cv: 'A switch puts someone else\'s position into the track, so the last two points can jump. CV reads that jump as speed.',
    ca: 'CA reads the same jump as a sudden acceleration and carries it forward with the square of the time ahead, so CA usually rises more than CV.',
    ade: 'ADE rises, but far less than with noise, because only the forecasts around a switch are hit.',
    fde: 'FDE rises by a similar percentage to ADE.',
  },
}
