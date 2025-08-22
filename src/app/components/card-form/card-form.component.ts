import { Component, EventEmitter, OnInit, Output } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { 
  Card, 
  ColorOptions, 
  CardTypeOptions, 
  SupertypeOptions,
  CommonSubtypes, 
  RarityOptions, 
  Rarity, 
  ManaSymbols,
  createEmptyCard 
} from '../../models/card.model';
import { ManaService } from '../../services/mana.service';

@Component({
  selector: 'app-card-form',
  templateUrl: './card-form.component.html',
  styleUrls: ['./card-form.component.scss']
})
export class CardFormComponent implements OnInit {
  cardForm!: FormGroup;
  colorOptions = ColorOptions;
  supertypeOptions = SupertypeOptions;
  cardTypeOptions = CardTypeOptions;
  rarityOptions = RarityOptions;
  manaSymbols = ManaSymbols;
  commonSubtypes = CommonSubtypes;
  
  showPowerToughness = false;
  filteredSubtypes: string[] = [];
  
  @Output() cardChange = new EventEmitter<Card>();
  @Output() generateCard = new EventEmitter<Card>();

  constructor(private fb: FormBuilder, private manaService: ManaService) {}

  ngOnInit(): void {
    this.initForm();
    
    // Emit initial values
    this.onFormValueChanges();
    
    // Subscribe to form changes to emit updates
    this.cardForm.valueChanges.subscribe(() => {
      this.onFormValueChanges();
    });

    // Subscribe to type changes to show/hide power/toughness
    this.cardForm.get('type')?.valueChanges.subscribe(type => {
      this.updateTypeRelatedFields(type);
    });
  }

  updateTypeRelatedFields(type: string): void {
    // Show power/toughness for creatures
    this.showPowerToughness = type?.toLowerCase().includes('creature') || false;
    
    // Update available subtypes based on main type
    this.updateFilteredSubtypes(type);
    
    // Update validators as needed
    if (this.showPowerToughness) {
      this.cardForm.get('power')?.setValidators([]);
      this.cardForm.get('toughness')?.setValidators([]);
    } else {
      this.cardForm.get('power')?.clearValidators();
      this.cardForm.get('toughness')?.clearValidators();
    }
    this.cardForm.get('power')?.updateValueAndValidity();
    this.cardForm.get('toughness')?.updateValueAndValidity();
  }

  updateFilteredSubtypes(type: string): void {
    if (!type) {
      this.filteredSubtypes = [];
      return;
    }
    
    // Find which main type the selected type belongs to
    const mainType = Object.keys(this.commonSubtypes).find(key => 
      type.toLowerCase().includes(key.toLowerCase())
    );
    
    this.filteredSubtypes = mainType 
      ? this.commonSubtypes[mainType as keyof typeof this.commonSubtypes] 
      : [];
  }

  initForm(): void {
    const emptyCard = createEmptyCard();
    
    this.cardForm = this.fb.group({
      name: [emptyCard.name, [Validators.maxLength(30)]],
      manaCost: [emptyCard.manaCost],
      supertype: [emptyCard.supertype],
      type: [emptyCard.type, [Validators.maxLength(50)]],
      subtype: [emptyCard.subtype],
      colors: [emptyCard.colors],
      cmc: [emptyCard.cmc, [Validators.min(0)]],
      rarity: [emptyCard.rarity],
      description: [emptyCard.description],
      power: [emptyCard.power],
      toughness: [emptyCard.toughness],
      powerToughness: [''], // Combined field for power/toughness
      flavorText: [emptyCard.flavorText],
      setCode: [emptyCard.setCode],
      cardNumber: [emptyCard.cardNumber]
    });
  }

  onFormValueChanges(): void {
    const formValue = this.cardForm.value;
    
    // If the type doesn't include 'creature', remove power/toughness
    if (!this.showPowerToughness) {
      formValue.power = undefined;
      formValue.toughness = undefined;
    }
    
    // Auto-generate art prompt
    formValue.artPrompt = this.generateArtPromptText();
    
    console.log('CardFormComponent: Emitting card changes with colors:', formValue.colors);
    console.log('CardFormComponent: Auto-generated art prompt:', formValue.artPrompt);
    this.cardChange.emit(formValue as Card);
  }

  onSubmit(): void {
    this.generateCard.emit(this.cardForm.value as Card);
  }

  isFieldInvalid(field: string): boolean {
    const control = this.cardForm.get(field);
    return !!control && control.invalid && (control.dirty || control.touched);
  }

  insertManaSymbol(symbol: string): void {
    const manaCostControl = this.cardForm.get('manaCost');
    if (manaCostControl) {
      const currentValue = manaCostControl.value || '';
      const newValue = currentValue + symbol;
      const reorderedValue = this.manaService.reorderManaSymbols(newValue);
      manaCostControl.setValue(reorderedValue);
      manaCostControl.markAsDirty();
      
      // Update colors and CMC after inserting mana symbol
      this.calculateCmcFromManaCost();
      this.updateColorsFromManaCost();
    }
  }

  clearManaCost(): void {
    this.cardForm.get('manaCost')?.setValue('');
    this.cardForm.get('manaCost')?.markAsDirty();
    
    // Update colors and CMC after clearing mana cost
    this.calculateCmcFromManaCost();
    this.updateColorsFromManaCost();
  }

  reorderManaCost(): void {
    const manaCostControl = this.cardForm.get('manaCost');
    if (manaCostControl) {
      const currentValue = manaCostControl.value || '';
      const reorderedValue = this.manaService.reorderManaSymbols(currentValue);
      if (reorderedValue !== currentValue) {
        manaCostControl.setValue(reorderedValue);
        manaCostControl.markAsDirty();
        
        // Update colors and CMC after reordering (just in case)
        this.calculateCmcFromManaCost();
        this.updateColorsFromManaCost();
      }
    }
  }

  onManaCostChange(): void {
    console.log('onManaCostChange called!');
    this.calculateCmcFromManaCost();
    this.updateColorsFromManaCost();
  }

  calculateCmcFromManaCost(): void {
    console.log('calculateCmcFromManaCost called');
    const manaCost = this.cardForm.get('manaCost')?.value || '';
    const cmc = this.manaService.calculateCMC(manaCost);
    
    // Update the CMC in the form if we have one
    if (this.cardForm.get('cmc')) {
      this.cardForm.get('cmc')?.setValue(cmc);
      this.cardForm.get('cmc')?.markAsDirty();
    }
  }


  updateColorsFromManaCost(): void {
    console.log('updateColorsFromManaCost called');
    const manaCost = this.cardForm.get('manaCost')?.value || '';
    const colorsFromMana = this.manaService.extractColorsFromManaCost(manaCost);
    
    console.log('Mana cost:', manaCost);
    console.log('Colors extracted from mana:', colorsFromMana);
    
    // Only use colors from mana cost, no merging
    this.cardForm.get('colors')?.setValue(colorsFromMana, { emitEvent: true });
    this.cardForm.get('colors')?.markAsDirty();
    
    console.log('Set colors to:', colorsFromMana);
    
    // Force emit the form changes immediately
    this.onFormValueChanges();
  }

  selectSupertype(supertype: string): void {
    this.cardForm.get('supertype')?.setValue(supertype);
    this.cardForm.get('supertype')?.markAsDirty();
  }

  selectCardType(type: string): void {
    this.cardForm.get('type')?.setValue(type);
    this.cardForm.get('type')?.markAsDirty();
  }

  selectSubtype(subtype: string): void {
    this.cardForm.get('subtype')?.setValue(subtype);
    this.cardForm.get('subtype')?.markAsDirty();
  }

  updateFullType(): void {
    const mainType = this.cardForm.get('type')?.value || '';
    const subtype = this.cardForm.get('subtype')?.value || '';
    
    if (mainType && subtype) {
      this.cardForm.get('type')?.setValue(`${mainType} — ${subtype}`);
    }
  }

  setRarity(rarity: Rarity): void {
    this.cardForm.get('rarity')?.setValue(rarity);
    this.cardForm.get('rarity')?.markAsDirty();
  }

  generateArtPromptText(): string {
    const name = this.cardForm.get('name')?.value || '';
    const supertype = this.cardForm.get('supertype')?.value || '';
    const type = this.cardForm.get('type')?.value || '';
    const subtype = this.cardForm.get('subtype')?.value || '';
    const colors = this.cardForm.get('colors')?.value || [];
    const rarity = this.cardForm.get('rarity')?.value || 'common';
    const power = this.cardForm.get('power')?.value || '';
    const toughness = this.cardForm.get('toughness')?.value || '';
    const description = this.cardForm.get('description')?.value || '';
    
    let prompt = 'Fantasy art of ';
    
    // Add name if available
    if (name) {
      prompt += name + ', ';
    }
    
    // Build type description
    let typeDescription = '';
    if (supertype) {
      typeDescription += supertype.toLowerCase() + ' ';
    }
    if (type) {
      typeDescription += type.toLowerCase();
    }
    if (subtype) {
      typeDescription += ' ' + subtype.toLowerCase();
    }
    
    if (typeDescription) {
      prompt += 'a ' + typeDescription + ', ';
    }
    
    // Add color-based atmospheric descriptions
    if (colors && colors.length > 0) {
      const colorDescriptions = [];
      if (colors.includes('W')) colorDescriptions.push('divine light and order');
      if (colors.includes('U')) colorDescriptions.push('arcane knowledge and control');
      if (colors.includes('B')) colorDescriptions.push('shadow and corruption');
      if (colors.includes('R')) colorDescriptions.push('chaotic fire and passion');
      if (colors.includes('G')) colorDescriptions.push('primal nature and growth');
      
      if (colorDescriptions.length > 0) {
        prompt += 'infused with ' + colorDescriptions.join(' and ') + ', ';
      }
    }
    
    // Add rarity-based grandeur
    if (rarity === 'mythic') {
      prompt += 'epic and legendary appearance, ';
    } else if (rarity === 'rare') {
      prompt += 'impressive and unique design, ';
    } else if (rarity === 'uncommon') {
      prompt += 'notable and interesting features, ';
    }
    
    // Add power level indication for creatures
    if (type && type.toLowerCase().includes('creature') && power && toughness) {
      const p = parseInt(power) || 0;
      const t = parseInt(toughness) || 0;
      const total = p + t;
      
      if (total >= 8) {
        prompt += 'massive and imposing, ';
      } else if (total >= 5) {
        prompt += 'strong and formidable, ';
      } else if (total >= 3) {
        prompt += 'agile and capable, ';
      } else {
        prompt += 'small but determined, ';
      }
    }
    
    // Extract key visual elements from description
    if (description) {
      const keyWords = description.split(' ').filter((word: string) =>
        word.length > 5 && 
        !['target', 'creature', 'player', 'opponent', 'enters', 'battlefield', 'combat'].includes(word.toLowerCase()) &&
        !word.includes('{') && !word.includes('}')
      ).slice(0, 2);
      
      if (keyWords.length > 0) {
        prompt += 'with elements of ' + keyWords.join(' and ') + ', ';
      }
    }
    
    // Add final artistic direction
    prompt += 'detailed digital art, Magic: The Gathering style';
    
    return prompt;
  }

  onColorChange(event: any, color: string): void {
    const currentColors = this.cardForm.get('colors')?.value || [];
    let newColors: string[];
    
    if (event.target.checked) {
      // Add color if not already present
      if (!currentColors.includes(color)) {
        newColors = [...currentColors, color];
      } else {
        newColors = currentColors;
      }
    } else {
      // Remove color
      newColors = currentColors.filter((c: string) => c !== color);
    }
    
    // Update the form control and force emit
    this.cardForm.get('colors')?.setValue(newColors, { emitEvent: true });
    this.cardForm.get('colors')?.markAsDirty();
    
    console.log('Manual color change:', color, event.target.checked ? 'added' : 'removed');
    console.log('Updated colors array:', newColors);
    
    // Force emit the form changes immediately
    this.onFormValueChanges();
  }

  getColorName(colorValue: string): string {
    const colorOption = this.colorOptions.find(option => option.value === colorValue);
    return colorOption ? colorOption.label : colorValue;
  }
  
  // Method to handle combined power/toughness input
  onPowerToughnessChange(): void {
    const powerToughnessValue = this.cardForm.get('powerToughness')?.value || '';
    const parts = powerToughnessValue.split('/');
    
    if (parts.length === 2) {
      const power = parts[0].trim();
      const toughness = parts[1].trim();
      
      this.cardForm.get('power')?.setValue(power);
      this.cardForm.get('toughness')?.setValue(toughness);
    }
  }
  
  // Update combined field when individual fields change
  updatePowerToughnessDisplay(): void {
    const power = this.cardForm.get('power')?.value || '';
    const toughness = this.cardForm.get('toughness')?.value || '';
    
    if (power && toughness) {
      this.cardForm.get('powerToughness')?.setValue(`${power}/${toughness}`, { emitEvent: false });
    }
  }
  
  // Convert mana symbol to CSS class for mana font
  getSymbolClass(symbol: string): string {
    return this.manaService.getSymbolClass(symbol);
  }

  // Get formatted symbol data for buttons
  getFormattedSymbolForButton(manaSymbol: any) {
    return this.manaService.formatSymbolForButton(manaSymbol);
  }
}
