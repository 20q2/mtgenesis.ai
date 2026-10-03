import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Card, Rarity } from '../../models/card.model';
import { CommanderFormComponent } from './commander-form.component';

describe('CommanderFormComponent', () => {
  let fixture: ComponentFixture<CommanderFormComponent>;
  let component: CommanderFormComponent;
  let last: Card;

  function setup(cmc = 3, taken: Partial<Record<Rarity, number>> = {}, initialCard: Partial<Card> | null = null) {
    TestBed.configureTestingModule({ declarations: [CommanderFormComponent] });
    fixture = TestBed.createComponent(CommanderFormComponent);
    component = fixture.componentInstance;
    component.cardChange.subscribe(card => (last = card));
    fixture.componentRef.setInput('cmc', cmc);
    fixture.componentRef.setInput('takenRarities', taken);
    fixture.componentRef.setInput('initialCard', initialCard);
    fixture.detectChanges();
  }

  const el = () => fixture.nativeElement as HTMLElement;
  const click = (selector: string) => {
    (el().querySelector(selector) as HTMLButtonElement).click();
    fixture.detectChanges();
  };

  afterEach(() => fixture.destroy());

  it('starts as a Legendary Creature with no rarity picked yet, auto body, colorless', () => {
    setup();
    expect(last.supertype).toBe('Legendary');
    expect(last.type).toBe('Creature');
    expect(last.commanderKind).toBe('creature');
    expect(last.rarity as string).toBe('');  // the player decides which commander gets which rarity
    expect(last.cmc).toBe(3);
    expect([last.power, last.toughness]).toEqual(['', '']);
    expect(last.manaCost).toBe('');
  });

  describe('colors', () => {
    it('adds colored pips and shows the full cost and mana value', () => {
      setup(4);
      click('.pip-add-W');
      click('.pip-add-U');
      expect(last.manaCost).toBe('{W}{U}');
      expect(last.colors).toEqual(['W', 'U']);
      expect(component.fullCost).toBe('{2}{W}{U}');
      expect(el().querySelector('.cost-value')!.textContent).toContain('4 mana');
    });

    it('offers no generic or X buttons', () => {
      setup();
      expect(el().querySelector('.pip-add-X')).toBeNull();
      expect(el().querySelector('.pip-add-1')).toBeNull();
      expect(el().querySelectorAll('.pip-add').length).toBe(6);
    });

    it('stops adding pips at 3 mana of pips', () => {
      setup();
      ['W', 'W', 'U'].forEach(c => click(`.pip-add-${c}`));
      expect((el().querySelector('.pip-add-B') as HTMLButtonElement).disabled).toBeTrue();
      click('.pip-add-B');
      expect(last.manaCost).toBe('{W}{W}{U}');
    });

    it("a 5 CMC commander's pips can fill all 5 mana", () => {
      setup(5);
      ['W', 'W', 'U', 'U', 'B'].forEach(c => click(`.pip-add-${c}`));
      expect(last.manaCost).toBe('{W}{W}{U}{U}{B}');
      expect(component.fullCost).toBe('{W}{W}{U}{U}{B}');
      expect((el().querySelector('.pip-add-R') as HTMLButtonElement).disabled).toBeTrue();
      expect(el().querySelector('.cost-value')!.textContent!.replace(/\s+/g, ' ')).toContain('5 of 5 mana colored');
    });

    it('says any colors can be combined', () => {
      setup(3);
      expect(el().querySelector('.cf-section .cf-rule')!.textContent).toContain('Any colors');
    });

    it('removes a pip when it is clicked in the cost, and clears all', () => {
      setup();
      ['W', 'U', 'B'].forEach(c => click(`.pip-add-${c}`));
      click('.cost-pip[data-index="1"]');
      expect(last.manaCost).toBe('{W}{B}');
      click('.cost-clear');
      expect(last.manaCost).toBe('');
    });
  });

  describe('kind', () => {
    it('Vehicle makes exactly a Legendary Artifact — Vehicle with 2 more points', () => {
      setup(3);
      component.subtype = 'Construct';
      click('.kind-vehicle');
      expect(last.type).toBe('Artifact');
      expect(last.subtype).toBe('Vehicle');
      expect(last.commanderKind).toBe('vehicle');
      expect(component.points).toBe(6);
      expect(el().querySelector('.subtype-chips')).toBeNull();
      expect(el().querySelector('#cf-subtype-3')).toBeNull();  // no type field: it is just Vehicle
      expect(el().querySelector('.vehicle-line')!.textContent).toContain('Legendary Artifact — Vehicle');
    });

    it('flags a creature type that is not a creature type', () => {
      setup(3);
      component.onSubtypeInput('Human Equipment');
      fixture.detectChanges();
      expect(component.subtypeError).toBe("Equipment isn't a creature type");
      expect(el().querySelector('.subtype-error')!.textContent).toContain("Equipment isn't a creature type");
    });

    it('Creature puts it back', () => {
      setup(3);
      click('.kind-vehicle');
      click('.kind-creature');
      expect(last.type).toBe('Creature');
      expect(last.subtype).not.toContain('Vehicle');
      expect(component.points).toBe(4);
    });
  });

  describe('body', () => {
    it('auto shows the split the server will pick', () => {
      setup(4);
      component.subtype = 'Goblin';
      component.emit();
      fixture.detectChanges();
      expect(component.shownStats).toEqual([3, 2]);
      expect(el().querySelector('.pt-value')!.textContent).toContain('3/2');
      expect(el().querySelector('.points-label')!.textContent).toContain('5 of 5 points');
    });

    it('a stepper switches to a typed body that can never exceed the points', () => {
      setup(3);
      // auto 2/2 already spends all 4 points, so + starts disabled
      expect((el().querySelector('.power-up') as HTMLButtonElement).disabled).toBeTrue();
      click('.toughness-down');
      expect(component.auto).toBeFalse();
      expect(component.shownStats).toEqual([2, 1]);
      click('.power-up');
      expect(component.shownStats).toEqual([3, 1]);
      expect([last.power, last.toughness]).toEqual(['3', '1']);
      expect(el().querySelector('.points-label')!.textContent).toContain('4 of 4 points');
      expect((el().querySelector('.power-up') as HTMLButtonElement).disabled).toBeTrue();
    });

    it('toughness never drops below 1 and power never below 0', () => {
      setup(3);
      click('.toughness-down');
      click('.toughness-down');
      expect(component.shownStats[1]).toBe(1);
      expect((el().querySelector('.toughness-down') as HTMLButtonElement).disabled).toBeTrue();
      click('.power-down'); click('.power-down'); click('.power-down');
      expect(component.shownStats[0]).toBe(0);
    });

    it('spending fewer points is allowed', () => {
      setup(3);
      click('.power-down');
      expect(component.shownStats).toEqual([1, 2]);
      expect(el().querySelector('.points-label')!.textContent).toContain('3 of 4 points');
    });

    it('Auto puts the body back to the server split', () => {
      setup(3);
      click('.power-down');
      click('.pt-auto');
      expect(component.auto).toBeTrue();
      expect([last.power, last.toughness]).toEqual(['', '']);
    });

    it('a Vehicle\'s typed body is clamped when it goes back to Creature', () => {
      setup(3);
      click('.kind-vehicle');
      click('.toughness-down');               // manual 3/2 of 6
      ['.power-up', '.power-up', '.power-up'].forEach(click);  // 6/... capped at 6 total
      click('.kind-creature');                // 4 points now
      const [p, t] = component.shownStats;
      expect(p + t).toBeLessThanOrEqual(4);
      expect(t).toBeGreaterThanOrEqual(1);
    });
  });

  describe('rarity', () => {
    it('offers Uncommon, Rare and Mythic', () => {
      setup();
      expect(Array.from(el().querySelectorAll('.rarity-gem')).map(b => b.textContent!.trim().split(/\s/)[0]))
        .toEqual(['Uncommon', 'Rare', 'Mythic']);
    });

    it('a rarity locked by another commander is disabled and says where', () => {
      setup(4, { uncommon: 3 });
      const uncommon = el().querySelector('.rarity-uncommon') as HTMLButtonElement;
      expect(uncommon.disabled).toBeTrue();
      expect(uncommon.textContent).toContain('Locked at 3 CMC');
      expect(last.rarity as string).toBe('');
    });

    it('clears a picked rarity that another commander then locks in', () => {
      setup(4);
      click('.rarity-uncommon');
      expect(last.rarity).toBe(Rarity.UNCOMMON);
      fixture.componentRef.setInput('takenRarities', { uncommon: 3 });
      fixture.detectChanges();
      expect(last.rarity as string).toBe('');
    });

    it('picks a rarity', () => {
      setup();
      click('.rarity-mythic');
      expect(last.rarity).toBe(Rarity.MYTHIC);
    });
  });

  it('fills in from a saved commander', () => {
    setup(4, {}, { manaCost: '{2}{W}{U}', type: 'Artifact', supertype: 'Legendary', subtype: 'Vehicle',
                   rarity: Rarity.MYTHIC, power: '5', toughness: '2', colors: ['W', 'U'], name: 'X', cmc: 4 });
    expect(last.manaCost).toBe('{W}{U}');
    expect(last.commanderKind).toBe('vehicle');
    expect(last.rarity).toBe(Rarity.MYTHIC);
    expect([last.power, last.toughness]).toEqual(['5', '2']);
    expect(component.subtype).toBe('');
  });

  it('previews the type line, cost, body and rarity', () => {
    setup(4);
    component.subtype = 'Human Cleric';
    click('.pip-add-B');
    click('.rarity-uncommon');
    const preview = el().querySelector('.preview')!.textContent!.replace(/\s+/g, ' ');
    expect(preview).toContain('Legendary Creature — Human Cleric');
    expect(preview).toContain('2/3');
    expect(preview).toContain('Uncommon');
  });

  it('draws the cost with mana-font classes', () => {
    setup(4);
    click('.pip-add-W');
    expect(component.symbolClass('{W}')).toBe('ms-w');
    const classes = Array.from(el().querySelectorAll('.preview-cost i')).map(i => i.className);
    expect(classes.some(c => c.includes('ms-3'))).toBeTrue();  // {3}{W} at 4 CMC
    expect(classes.some(c => c.includes('ms-w'))).toBeTrue();
  });

  it('sizes the art prompt by CMC', () => {
    setup(5);
    component.subtype = 'Dragon';
    component.emit();
    expect(last.artPrompt).toBe('a legendary dragon, large and imposing');
  });
});
