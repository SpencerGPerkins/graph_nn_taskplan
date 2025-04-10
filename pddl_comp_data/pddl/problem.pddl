(define (problem lock-red-wire)
  (:domain robot-arm)
  (:objects
    arm1 arm2 - robot
    red_wire - wire
    green_wire - wire
    green_wire - wire
    green_wire - wire
    red_wire - wire
  )
  (:init
    (arm-empty arm1)
    (arm-empty arm2)
    (is-arm1 arm1)
    (is-arm2 arm2)
    (on_table red_wire)
    (on_table green_wire)
    (on_table green_wire)
    (inserted green_wire)
    (on_table red_wire)
  )
  (:goal
    (locked red_wire)
  )
)


