(define (domain robot-arm)
  (:requirements :typing :equality :strips)

  (:types
    wire robot location ; location type is still kept for potential future expansions
  )

  (:predicates
    (holding ?wire - wire)
    (on_table ?wire - wire)
    (locked ?wire - wire)
    (inserted ?wire - wire)
    (arm-empty ?arm - robot)
    (is-arm2 ?arm - robot)
    (is-arm1 ?arm - robot)
  )

  (:action pickup
    :parameters
      (?arm - robot
       ?wire - wire)
    :precondition
      (and
        (on_table ?wire)
        (arm-empty ?arm)
      )
    :effect
      (and
        (not (on_table ?wire))
        (holding ?wire)
        (not (arm-empty ?arm))
      )
  )

  (:action putdown
    :parameters
      (?arm - robot
       ?wire - wire)
    :precondition
      (and
        (holding ?wire)
        (is-arm1 ?arm)
      )
    :effect
      (and
        (on_table ?wire)
        (arm-empty ?arm)
        (not (holding ?wire))
      )
  )

  (:action lock
    :parameters
      (?arm - robot
       ?wire - wire)
    :precondition
      (and
        (inserted ?wire)
        (is-arm2 ?arm)
      )
    :effect
      (and
        (locked ?wire)
        (arm-empty ?arm)
        (not (inserted ?wire))
      )
  )

  (:action insert
    :parameters
      (?arm - robot
       ?wire - wire)
    :precondition
      (and
        (holding ?wire)
        (is-arm1 ?arm)
      )
    :effect
      (and
        (inserted ?wire)
        (not (holding ?wire))
      )
  )
)
