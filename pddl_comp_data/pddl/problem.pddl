(define (problem lock-red-wire)
  (:domain robot-arm)
  (:objects
    arm1 arm2 - robot
    red_wire - wire
  )
  (:init
    (arm-empty arm1)
    (arm-empty arm2)
    (is-arm1 arm1)
    (is-arm2 arm2)
    (on_table red_wire)
  )
  (:goal
    (locked red_wire)
  )
)


